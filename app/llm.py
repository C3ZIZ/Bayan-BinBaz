from functools import lru_cache
from typing import List, Dict, Any

from llama_cpp import Llama


@lru_cache(maxsize=1)
def get_llm() -> Llama:
    """
    Load ALLaM-7B-Instruct (GGUF) via llama-cpp-python.
    """
    llm = Llama.from_pretrained(
        repo_id="Omartificial-Intelligence-Space/ALLaM-7B-Instruct-preview-Q4_K_M-GGUF",
        filename="*q4_k_m.gguf",
        n_ctx=4096,
        n_gpu_layers=0,  # If CUDA is available, use GPU; otherwise CPU
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


def build_exact_prompt(user_question: str, hit: Dict[str, Any]) -> str:
    return f"""
السؤال من المستخدم:
{user_question}

أقرب سؤال مطابق في قاعدة بيانات الشيخ ابن باز:
السؤال:
{hit.get("question", "").strip()}

الجواب:
{hit.get("answer", "").strip()}

المطلوب:
- أعد عرض الجواب للمستخدم بلغة واضحة ومبسطة مع الحفاظ على نفس الحكم الشرعي.
- يمكنك تلخيص الشرح أو إعادة ترتيبه، لكن لا تغيّر المعنى.
- لا تذكر رابط الفتوى أو رقمها لأن النظام الخارجي سيعرضها للمستخدم.

أجب في فقرتين على الأكثر.
""".strip()


def build_approx_prompt(user_question: str, hits: List[Dict[str, Any]]) -> str:
    context_parts = []
    for i, h in enumerate(hits, start=1):
        part = f"""فتوى رقم {i}:
السؤال: {h.get("question", "").strip()}
الجواب: {h.get("answer", "").strip()}
"""
        context_parts.append(part)
    context_text = "\n\n".join(context_parts)

    return f"""
السؤال من المستخدم (لا يوجد له تطابق تام في القاعدة):
{user_question}

فيما يلي مجموعة فتاوى قريبة في الموضوع للشيخ ابن باز:

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
        # ما فيه سياق نهائيًا
        user_prompt = f"""
السؤال:
{user_question}

لم أجد أي فتوى مرتبطة بهذا السؤال في قاعدة بيانات الشيخ ابن باز.
رجاءً اجِب بإرشاد عام، واذكر بوضوح أن المستخدم يجب أن يسأل أهل العلم مباشرة.
        """.strip()
    else:
        if exact:
            user_prompt = build_exact_prompt(user_question, hits[0])
        else:
            user_prompt = build_approx_prompt(user_question, hits)

    result = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        max_tokens=768,
    )
    answer = result["choices"][0]["message"]["content"]
    return answer.strip()
