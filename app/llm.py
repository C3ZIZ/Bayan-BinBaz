from functools import lru_cache # also store ALLAM chace
from typing import List, Dict, Any
import threading

from llama_cpp import Llama # GGUF runner

# Got problem with HuggingFace managing this model locally.
MAX_ANSWER_CHARS = 800
MAX_HITS_FOR_PROMPT = 3
MAX_TOKENS = 384

_LLM_LOCK = threading.Lock()


@lru_cache(maxsize=1)
def get_llm() -> Llama:
    llm = Llama.from_pretrained(
        repo_id="Omartificial-Intelligence-Space/ALLaM-7B-Instruct-preview-Q4_K_M-GGUF",
        filename="*q4_k_m.gguf",
        n_ctx=2048,
        n_gpu_layers=0,
        n_batch=64,
        n_threads=4,
        use_mmap=True,
        use_mlock=False,
        verbose=False,
    )
    return llm


SYSTEM_PROMPT = """
أنت مساعد افتراضي متخصص في فتاوى سماحة الشيخ عبد العزيز بن باز رحمه الله فقط.
تعتمد إجاباتك على الفتاوى التي يزوّدك بها النظام في نص السياق.
التزم بالآتي:
- لا تُصدر أحكامًا جديدة من عندك، بل استخرج الحكم من نصوص الفتاوى فقط.
- إذا لم يكفِ السياق لإعطاء جواب واضح، وضّح للمستخدم أن الجواب تقريبي وأن عليه الرجوع إلى عالم موثوق.
- أجب باللغة العربية الفصحى المبسّطة.
- اختم كل جواب بتنبيه مثل: «هذا الجواب آلي مبني على فتاوى الشيخ ابن باز، ولا يغني عن سؤال أهل العلم مباشرة».
""".strip()


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


def generate_answer(
    user_question: str,
    hits: List[Dict[str, Any]],
    exact: bool,
) -> str:
    llm = get_llm()

    if not hits:
        user_prompt = f"""
السؤال:
{user_question}

لم أجد أي فتوى مرتبطة بهذا السؤال في قاعدة بيانات الشيخ ابن باز.
رجاءً:
- قدّم توجيهًا عامًا جدًا إن كان في قدرتك، بدون إصدار حكم تفصيلي.
- اذكر بوضوح أن هذه ليست فتوى عن الشيخ ابن باز، وأن على السائل أن يسأل أهل العلم مباشرة.
        """.strip()
    else:
        if exact:
            user_prompt = build_exact_prompt(user_question, hits[0])
        else:
            user_prompt = build_approx_prompt(user_question, hits)

    with _LLM_LOCK:
        result = llm.create_chat_completion(
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=MAX_TOKENS,
        )

    answer = result["choices"][0]["message"]["content"]
    return answer.strip()
