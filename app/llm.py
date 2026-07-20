import os
from functools import lru_cache
from typing import List, Dict, Any, Iterator

from huggingface_hub import InferenceClient

# ---------------------------------------------------------------------------
# LLM backend: Hugging Face Inference API (chat completion, streaming).
#
# Rationale: on a small CPU VPS (<=4GB RAM) a local 7B GGUF cannot run. Moving
# the LLM to a hosted API gives top-tier Arabic answers with ZERO local LLM RAM
# and native token streaming (used by the /api/chat/stream endpoint). BGE-M3
# retrieval still runs locally. The old local llama.cpp path is kept for
# reference in app/archive/llm.py.
#
# Everything is env-configurable so the same image runs anywhere.
# ---------------------------------------------------------------------------

MAX_ANSWER_CHARS = int(os.getenv("LLM_CONTEXT_CHARS", "800"))  # أقصى طول مقتطف الفتوى في البرومبت
MAX_HITS_FOR_PROMPT = int(os.getenv("LLM_MAX_HITS", "3"))      # عدد الفتاوى في وضع approx
MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "512"))           # أقصى عدد توكينات في الجواب
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))

# Any chat model reachable by your HF token (HF Inference Providers). Override
# via LLM_API_MODEL. Default is a strong multilingual model with good Arabic.
LLM_API_MODEL = os.getenv("LLM_API_MODEL", "Qwen/Qwen2.5-72B-Instruct")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "auto")  # "auto" lets HF route to an available provider
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "120"))
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACEHUB_API_TOKEN")


SYSTEM_PROMPT = """
أنت مساعد افتراضي متخصص في فتاوى سماحة الشيخ عبد العزيز بن باز رحمه الله فقط.
تعتمد إجاباتك على الفتاوى التي يزوّدك بها النظام في نص السياق.
التزم بالآتي:
- لا تُصدر أحكامًا جديدة من عندك، بل استخرج الحكم من نصوص الفتاوى فقط.
- إذا لم يكفِ السياق لإعطاء جواب واضح، وضّح للمستخدم أن الجواب تقريبي وأن عليه الرجوع إلى عالم موثوق.
- أجب باللغة العربية الفصحى المبسّطة.
- اختم كل جواب بتنبيه مثل: «هذا الجواب آلي مبني على فتاوى الشيخ ابن باز، ولا يغني عن سؤال أهل العلم مباشرة».
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
            model=LLM_API_MODEL,
            token=HF_TOKEN,
            provider=LLM_PROVIDER,
            timeout=LLM_TIMEOUT,
        )
    except TypeError:
        # Older huggingface_hub without the `provider` kwarg.
        return InferenceClient(model=LLM_API_MODEL, token=HF_TOKEN, timeout=LLM_TIMEOUT)


def _truncate(text: str, max_chars: int = MAX_ANSWER_CHARS) -> str:
    if not text:
        return ""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def build_exact_prompt(user_question: str, hit: Dict[str, Any]) -> str:
    q = (hit.get("question") or "").strip()
    a = _truncate(hit.get("answer") or "")

    return f"""
السؤال من المستخدم:
{user_question}

أقرب سؤال مطابق في قاعدة بيانات الشيخ ابن باز:
السؤال:
{q}

الجواب (مقتطف من الفتوى):
{a}

المطلوب:
- أعد عرض الجواب للمستخدم بلغة واضحة ومبسطة مع الحفاظ على نفس الحكم الشرعي.
- يمكنك تلخيص الشرح أو إعادة ترتيبه، لكن لا تغيّر المعنى.
- لا تذكر رابط الفتوى أو رقمها لأن النظام الخارجي سيعرضها للمستخدم.

أجب في فقرتين على الأكثر.
""".strip()


def build_approx_prompt(user_question: str, hits: List[Dict[str, Any]]) -> str:
    hits = hits[:MAX_HITS_FOR_PROMPT]

    parts = []
    for i, h in enumerate(hits, start=1):
        q = (h.get("question") or "").strip()
        a = _truncate(h.get("answer") or "")
        parts.append(
            f"""فتوى رقم {i}:
السؤال: {q}
الجواب (مقتطف): {a}
"""
        )

    context_text = "\n\n".join(parts)

    return f"""
السؤال من المستخدم (لا يوجد له تطابق تام في القاعدة):
{user_question}

فيما يلي مقتطفات من فتاوى قريبة للشيخ ابن باز:

{context_text}

المطلوب:
- استخرج من الفتاوى السابقة ما يساعد على توجيه السائل.
- إن كان الحكم غير واضح، فاذكر أن الجواب تقريبي، وأن عليه أن يسأل عالمًا موثوقًا.
- لا تُخترع أحكامًا جديدة، وتجنّب الخوض في ما لا تغطيه الفتاوى السابقة.

أجب في فقرة أو فقرتين بالعربية الفصحى المبسطة.
""".strip()


def build_user_prompt(user_question: str, hits: List[Dict[str, Any]], exact: bool) -> str:
    if not hits:
        return f"""
السؤال:
{user_question}

لم أجد أي فتوى مرتبطة بهذا السؤال في قاعدة بيانات الشيخ ابن باز.
رجاءً:
- قدّم توجيهًا عامًا جدًا إن كان في قدرتك، بدون إصدار حكم تفصيلي.
- اذكر بوضوح أن هذه ليست فتوى عن الشيخ ابن باز، وأن على السائل أن يسأل أهل العلم مباشرة.
""".strip()
    if exact:
        return build_exact_prompt(user_question, hits[0])
    return build_approx_prompt(user_question, hits)


def stream_answer(
    user_question: str,
    hits: List[Dict[str, Any]],
    exact: bool,
) -> Iterator[str]:
    """Yield answer text deltas as they arrive from the API (for SSE streaming)."""
    client = get_client()
    user_prompt = build_user_prompt(user_question, hits, exact)

    stream = client.chat_completion(
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
    user_question: str,
    hits: List[Dict[str, Any]],
    exact: bool,
) -> str:
    """Non-streaming convenience wrapper (used by POST /api/chat)."""
    return "".join(stream_answer(user_question, hits, exact)).strip()
