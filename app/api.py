from fastapi import APIRouter
from .schemas import ChatRequest, ChatResponse, FatwaHit
from .retrieval import get_retriever
from .llm import generate_answer



router = APIRouter()


EXACT_THRESHOLD = 0.90


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    retriever = get_retriever()
    hits_raw = retriever.search(request.question, top_k=request.top_k)

    if not hits_raw:
        llm_answer = generate_answer(request.question, [], exact=False)
        header = (
            "الحالة: لا يوجد أي فتوى قريبة لهذا السؤال في قاعدة بيانات الشيخ ابن باز.\n"
            "تنبيه: هذه إجابة عامة، ويجب الرجوع لأهل العلم مباشرة."
        )
        answer = header + "\n\n" + llm_answer
        return ChatResponse(
            mode="none",
            exact_match=False,
            similarity=0.0,
            answer=answer,
            related_fatwas=[],
        )

    best = hits_raw[0]
    exact = best["similarity"] >= EXACT_THRESHOLD

    llm_answer = generate_answer(request.question, hits_raw, exact=exact)

    if exact:
        header = "الحالة: تم العثور على سؤال مطابق تقريبًا في قاعدة بيانات الشيخ ابن باز."
    else:
        header = (
            "الحالة: لا يوجد سؤال مطابق حرفيًا في القاعدة؛ "
            "هذه إجابة تقريبية مبنية على أقرب الفتاوى المتاحة."
        )

    answer = header + "\n\n" + llm_answer

    related = [
        FatwaHit(
            id=h["id"],
            question=h["question"],
            title=h["title"],
            link=h["link"],
            similarity=h["similarity"],
            categories=h.get("categories") if isinstance(h.get("categories"), list) else None,
        )
        for h in hits_raw
    ]

    return ChatResponse(
        mode="exact" if exact else "approx",
        exact_match=exact,
        similarity=best["similarity"],
        answer=answer,
        matched_question=best["question"] if exact else None,
        fatwa_link=best["link"] if exact else None,
        related_fatwas=related,
    )
