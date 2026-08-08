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


def format_sources(sources: List[Dict[str, Any]]) -> str:
    """Render the gate-approved fatwas as a numbered list the model cites into."""
    blocks = []
    for i, s in enumerate(sources, start=1):
        blocks.append(
            f"[{i}]\n"
            f"السؤال: {(s.get('question') or '').strip()}\n"
            f"جواب الشيخ: {truncate_at_sentence(s.get('answer') or '', MAX_ANSWER_CHARS)}"
        )
    return "\n\n".join(blocks)


def build_direct_prompt(question: str, sources: List[Dict[str, Any]]) -> str:
    return f"""
سؤال المستخدم:
{question}

فتاوى الشيخ ابن باز المتعلقة بالسؤال:

{format_sources(sources)}

المطلوب:
- اعرض الحكم كما قرره الشيخ، بلغة واضحة مبسطة، دون تغيير المعنى.
- ضع بعد كل حكم رقم الفتوى التي أخذته منها هكذا [1].
- لا تضف حكمًا ليس في الفتاوى أعلاه.

أجب في فقرتين على الأكثر.
""".strip()


def build_derived_prompt(question: str, sources: List[Dict[str, Any]]) -> str:
    """The common case. The structural separation demanded here is what keeps a
    derived application from being read as a direct ruling by the shaykh."""
    return f"""
سؤال المستخدم:
{question}

لا توجد فتوى للشيخ ابن باز على هذه الحالة بعينها، لكن الفتاوى الآتية تتناول أصل المسألة:

{format_sources(sources)}

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
) -> str:
    if verdict == "direct":
        return build_direct_prompt(question, sources)
    if verdict == "derived":
        return build_derived_prompt(question, sources)
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
    client = get_client()
    stream = client.chat_completion(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_user_prompt(question, sources, verdict, premise_issue),
            },
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
