"""LLM layer: the answerability gate's transport, and grounded answer generation.

Design rule: the system's default is NOT to assert. An answer is a claim about
what Shaykh Ibn Baz ruled, so every such claim must carry a ``[n]`` marker
pointing at a fatwa the system actually retrieved. Generation only ever sees the
fatwas the gate approved (app/gate.py), so the model cannot cite what it was
never given.

The ``derived`` path is the common case, not a degraded one: a fatwa's own
question self-matches at a median cosine of only 0.815, so treating "no exact
match" as "no answer" would refuse most legitimate questions. Its prompt is
stricter than the direct path in one specific way — it must visibly separate
what the shaykh stated from the application of that ruling to the asker's case.
Presenting a derived application as a direct ruling from Ibn Baz is
misattribution, the worst failure this system can commit.
"""

import os
from functools import lru_cache
from typing import Any, Dict, Iterator, List, Optional

from huggingface_hub import InferenceClient

from .textutil import truncate_at_sentence

MAX_ANSWER_CHARS = int(os.getenv("LLM_CONTEXT_CHARS", "1400"))
MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "900"))
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))

# Which LLM serves requests:
#   "api"   — Hugging Face Inference only. Best quality, metered; a depleted
#             quota takes the app down.
#   "local" — GGUF model in-process only. No quota, no per-question cost.
#   "both"  — API first, local as FALLBACK when the API fails (402 out of
#             credits, 429 throttled, timeout, provider outage). THE DEFAULT:
#             normal traffic gets the large hosted model, and an outage degrades
#             to a smaller local one instead of taking the app down.
#
# Because "both" is the default, an install with no HF_TOKEN still works — the
# API call fails immediately and the local model serves the request. The first
# such request downloads ~1.9GB, so it is slow; pre-pull the weights or set
# LLM_BACKEND=api if you would rather fail fast than fall back.
#
# See app/llm_local.py for the benchmark behind the default local model.
LLM_BACKEND = os.getenv("LLM_BACKEND", "both").strip().lower()

# The local model may run the GATE (it scored 5/5 there) but must not write a
# ruling: measured, it fabricated "praying Isha as 5 rakʿahs is permissible"
# from a fatwa about voluntary prayer, complete with valid citations.
LOCAL_ALLOW_RULINGS = os.getenv("LOCAL_ALLOW_RULINGS", "0") == "1"

_USE_API = LLM_BACKEND in ("api", "both")
_USE_LOCAL = LLM_BACKEND in ("local", "both")
_FALLBACK = LLM_BACKEND == "both"

LLM_API_MODEL = os.getenv("LLM_API_MODEL", "Qwen/Qwen2.5-72B-Instruct")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "auto")
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "120"))
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")

DISCLAIMER = (
    "هذا الجواب آلي مبني على فتاوى الشيخ ابن باز، ولا يغني عن سؤال أهل العلم مباشرة."
)

SYSTEM_PROMPT = f"""
أنت مساعد يعرض فتاوى سماحة الشيخ عبد العزيز بن باز رحمه الله، ولست مفتيًا.

القاعدة الأولى وهي فوق كل ما سواها:
- لا تُقرِّر حكمًا شرعيًا إلا وهو مأخوذ من فتوى معروضة عليك في السياق.
- كل حكم تذكره يجب أن يتبعه رقم مرجعه هكذا [1] أو [2].
- إن لم تجد في الفتاوى المعروضة ما يُثبت الحكم فقل ذلك صراحةً، ولا تستنبط من عندك.
- لا تذكر رقمًا لم يُعرض عليك.

وأيضًا:
- أجب بالعربية الفصحى المبسّطة.
- لا تذكر روابط الفتاوى؛ النظام يعرضها للمستخدم.
- اختم كل جواب بهذا التنبيه: «{DISCLAIMER}»
""".strip()


@lru_cache(maxsize=1)
def get_client() -> InferenceClient:
    """Cached Hugging Face Inference client (no local weights loaded)."""
    if not HF_TOKEN:
        raise RuntimeError(
            "HF_TOKEN is not set. The LLM runs via the Hugging Face Inference API; "
            "set HF_TOKEN (or HUGGINGFACEHUB_API_TOKEN) in the environment."
        )
    try:
        return InferenceClient(
            model=LLM_API_MODEL, token=HF_TOKEN, provider=LLM_PROVIDER, timeout=LLM_TIMEOUT
        )
    except TypeError:  # older huggingface_hub without the `provider` kwarg
        return InferenceClient(model=LLM_API_MODEL, token=HF_TOKEN, timeout=LLM_TIMEOUT)


def chat_completion(
    system: str,
    user: str,
    model: Optional[str] = None,
    max_tokens: int = 512,
    temperature: float = 0.0,
) -> str:
    """Blocking chat completion. Used by the gate, which must finish before
    generation starts."""
    if _USE_LOCAL and not _USE_API:
        return _local_chat(system, user, model, max_tokens, temperature)

    try:
        return _api_chat(system, user, model, max_tokens, temperature)
    except Exception as exc:
        if not _FALLBACK:
            raise
        print(f"[llm] API failed ({type(exc).__name__}: {exc}); falling back to local.")
        return _local_chat(system, user, model, max_tokens, temperature)


def _local_chat(system, user, model, max_tokens, temperature) -> str:
    from . import llm_local

    return llm_local.chat_completion(
        system, user, model, max_tokens=max_tokens, temperature=temperature
    )


def _api_chat(system, user, model, max_tokens, temperature) -> str:
    result = get_client().chat_completion(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        model=model or LLM_API_MODEL,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return (result.choices[0].message.content or "").strip()


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def format_sources(
    sources: List[Dict[str, Any]], max_chars: int = None, max_sources: int = None
) -> str:
    """Render the gate-approved fatwas as a numbered list the model cites into.

    The budget is adjustable because prompt PREFILL dominates cost on a CPU
    local model: measured, the same model runs ~4 tok/s on a short prompt but
    ~1.2 tok/s once ~1600 tokens of fatwa text are prepended.
    """
    limit = MAX_ANSWER_CHARS if max_chars is None else max_chars
    if max_sources is not None:
        sources = sources[:max_sources]
    blocks = []
    for i, s in enumerate(sources, start=1):
        blocks.append(
            f"[{i}]\n"
            f"السؤال: {(s.get('question') or '').strip()}\n"
            f"جواب الشيخ: {truncate_at_sentence(s.get('answer') or '', limit)}"
        )
    return "\n\n".join(blocks)


def build_direct_prompt(
    question: str,
    sources: List[Dict[str, Any]],
    max_chars: int = None,
    max_sources: int = None,
) -> str:
    return f"""
سؤال المستخدم:
{question}

فتاوى الشيخ ابن باز المتعلقة بالسؤال:

{format_sources(sources, max_chars, max_sources)}

المطلوب:
- اعرض الحكم كما قرره الشيخ، بلغة واضحة مبسطة، دون تغيير المعنى.
- ضع بعد كل حكم رقم الفتوى التي أخذته منها هكذا [1].
- لا تضف حكمًا ليس في الفتاوى أعلاه.

أجب في فقرتين على الأكثر.
""".strip()


def build_derived_prompt(
    question: str,
    sources: List[Dict[str, Any]],
    max_chars: int = None,
    max_sources: int = None,
) -> str:
    """The common case. The structural separation demanded here is what keeps a
    derived application from being read as a direct ruling by the shaykh."""
    return f"""
سؤال المستخدم:
{question}

لا توجد فتوى للشيخ ابن باز على هذه الحالة بعينها، لكن الفتاوى الآتية تتناول أصل المسألة:

{format_sources(sources, max_chars, max_sources)}

المطلوب، والتزم هذا الترتيب:
1. ابدأ بجملة تُوضّح أنه لا توجد فتوى على هذه الحالة بعينها.
2. اذكر ما قرره الشيخ صراحةً في الفتاوى أعلاه، وضع مرجع كل حكم هكذا [1].
3. ثم بيّن كيف ينطبق ذلك على حالة السائل، وابدأ هذا الجزء بعبارة مثل «وعليه، فإن...».
   لا تُدخل في هذا الجزء أي حكم لم يرد في الفتاوى أعلاه.
4. اختم بنصيحة السائل بسؤال أهل العلم لخصوص حالته.

لا تنسب إلى الشيخ ما لم يقله. أجب في ثلاث فقرات على الأكثر.
""".strip()


def build_abstain_prompt(question: str, premise_issue: str) -> str:
    if premise_issue:
        return f"""
سؤال المستخدم:
{question}

هذا السؤال مبني على افتراض غير صحيح. بيان الخلل:
{premise_issue}

المطلوب:
- وضّح للسائل بلطف أن السؤال قائم على افتراض لا يقع، واشرح سبب ذلك بإيجاز.
- لا تُصدر أي حكم شرعي، ولا تذكر أي فتوى، ولا تفترض ما لم يسأل عنه.
- إن كان يقصد شيئًا آخر فادعُه إلى إعادة صياغة سؤاله.

أجب في فقرة واحدة قصيرة.
""".strip()

    return f"""
سؤال المستخدم:
{question}

لم أجد في فتاوى الشيخ ابن باز ما يُثبت جوابًا لهذا السؤال.

المطلوب:
- أخبر السائل بوضوح أنه لا توجد في قاعدة فتاوى الشيخ ابن باز فتوى تجيب على سؤاله.
- لا تُصدر حكمًا شرعيًا من عندك، ولا تُخمّن.
- وجّهه إلى سؤال أهل العلم الموثوقين.

أجب في فقرة واحدة قصيرة.
""".strip()


def build_user_prompt(
    question: str,
    sources: List[Dict[str, Any]],
    verdict: str,
    premise_issue: str = "",
    max_chars: int = None,
    max_sources: int = None,
) -> str:
    if verdict == "direct":
        return build_direct_prompt(question, sources, max_chars, max_sources)
    if verdict == "derived":
        return build_derived_prompt(question, sources, max_chars, max_sources)
    return build_abstain_prompt(question, premise_issue)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def stream_answer(
    question: str,
    sources: List[Dict[str, Any]],
    verdict: str,
    premise_issue: str = "",
) -> Iterator[str]:
    """Yield answer text deltas as they arrive (for SSE streaming).

    ``sources`` must already be filtered to the gate-approved fatwas — this
    function does not re-check relevance.
    """
    user_prompt = build_user_prompt(question, sources, verdict, premise_issue)

    def local_refusal() -> Iterator[str]:
        """What the local model is allowed to say about a ruling: nothing.

        Measured on the real app, the 3B local model answered «هل اقدر اصلي
        العشاء ٥ ركعات؟» with "yes, permissible" — a fabricated ruling on an
        obligatory prayer, derived by misapplying a fatwa about VOLUNTARY
        prayer, and delivered with valid citations. The citation validator
        cannot catch that: the markers are real, the inference is false.

        So when the hosted model is unavailable, the system reports that and
        points at the retrieved fatwas instead of ruling from a small model.
        Set LOCAL_ALLOW_RULINGS=1 to override, knowing the above.
        """
        yield (
            "تعذّر الوصول إلى النموذج الأساسي حاليًا، ولذلك لن يصدر هذا النظام "
            "حكمًا شرعيًا الآن؛ فالنموذج الاحتياطي المحلي أصغر من أن يُعتمد عليه "
            "في الفتوى.\n\n"
        )
        if sources:
            yield "وهذه أقرب فتاوى الشيخ ابن باز صلةً بسؤالك، وفيها الجواب بإذن الله:\n"
            for i, src in enumerate(sources, start=1):
                title = (src.get("title") or src.get("question") or "فتوى").strip()
                yield f"\n{i}. {title}\n{src.get('link', '')}\n"
            yield "\nراجعها مباشرة، أو أعد المحاولة بعد قليل.\n"
        else:
            yield "أعد المحاولة بعد قليل، أو اسأل أهل العلم مباشرة.\n"

    def local_prompt() -> str:
        """A tighter prompt for the local model — prefill is its dominant cost."""
        from . import llm_local

        return build_user_prompt(
            question, sources, verdict, premise_issue,
            max_chars=llm_local.LOCAL_CONTEXT_CHARS,
            max_sources=llm_local.LOCAL_MAX_SOURCES,
        )

    if _USE_LOCAL and not _USE_API:
        if verdict in ("direct", "derived") and not LOCAL_ALLOW_RULINGS:
            yield from local_refusal()
            return
        yield from _local_stream(local_prompt())
        return

    # Fallback is only safe BEFORE the first token reaches the client: once text
    # has been streamed it cannot be retracted, and silently continuing from a
    # different model would splice two answers together.
    emitted = False
    try:
        for delta in _api_stream(user_prompt):
            emitted = True
            yield delta
    except Exception as exc:
        if emitted or not _FALLBACK:
            raise
        print(f"[llm] API failed ({type(exc).__name__}: {exc}); falling back to local.")
        if verdict in ("direct", "derived") and not LOCAL_ALLOW_RULINGS:
            yield from local_refusal()
            return
        yield from _local_stream(local_prompt())


def _local_stream(user_prompt: str) -> Iterator[str]:
    from . import llm_local

    yield from llm_local.stream_completion(
        SYSTEM_PROMPT, user_prompt,
        max_tokens=llm_local.LOCAL_MAX_TOKENS, temperature=TEMPERATURE,
    )


def _api_stream(user_prompt: str) -> Iterator[str]:
    stream = get_client().chat_completion(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        stream=True,
    )
    for chunk in stream:
        try:
            delta = chunk.choices[0].delta.content
        except (AttributeError, IndexError, KeyError):
            delta = None
        if delta:
            yield delta


def generate_answer(
    question: str,
    sources: List[Dict[str, Any]],
    verdict: str,
    premise_issue: str = "",
) -> str:
    """Non-streaming convenience wrapper (used by POST /api/chat)."""
    return "".join(stream_answer(question, sources, verdict, premise_issue)).strip()


def describe_backend() -> dict:
    """Which LLM is actually serving requests. Surfaced by /health."""
    info = {"backend": LLM_BACKEND, "fallback_to_local": _FALLBACK}
    if _USE_API:
        info["api"] = {"model": LLM_API_MODEL, "provider": LLM_PROVIDER,
                       "token_configured": bool(HF_TOKEN)}
    if _USE_LOCAL:
        from . import llm_local

        info["local"] = llm_local.describe()
    return info
