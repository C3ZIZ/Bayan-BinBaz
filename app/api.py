import json
from typing import Any, Dict, List, Optional, Tuple

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from .retrieval import get_retriever
from .llm import generate_answer, stream_answer
from .schemas import ChatRequest, ChatResponse, FatwaHit


router = APIRouter()


EXACT_THRESHOLD = 0.90  # عتبة الـ cosine similarity لاعتبار السؤال "نفسه تقريبًا"


def _header_for(mode: str) -> str:
    if mode == "none":
        return (
            "الحالة: لا يوجد أي فتوى قريبة لهذا السؤال في قاعدة بيانات الشيخ ابن باز.\n"
            "تنبيه: هذه إجابة عامة، ويجب الرجوع لأهل العلم مباشرة."
        )
    if mode == "exact":
        return "الحالة: تم العثور على سؤال مطابق تقريبًا في قاعدة بيانات الشيخ ابن باز."
    return (
        "الحالة: لا يوجد سؤال مطابق حرفيًا في القاعدة؛ "
        "هذه إجابة تقريبية مبنية على أقرب الفتاوى المتاحة."
    )


def _categories(value: Any) -> Optional[List[str]]:
    if isinstance(value, list):
        return value
    if hasattr(value, "tolist"):  # numpy array
        try:
            return list(value.tolist())
        except Exception:
            return None
    return None


def _related(hits_raw: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "id": h["id"],
            "question": h["question"],
            "title": h["title"],
            "link": h["link"],
            "similarity": h["similarity"],
            "categories": _categories(h.get("categories")),
        }
        for h in hits_raw
    ]


def _retrieve(request: ChatRequest) -> Tuple[List[Dict[str, Any]], bool, str]:
    """Run retrieval and derive (hits, exact, mode)."""
    retriever = get_retriever()
    hits_raw = retriever.search(request.question, top_k=request.top_k)
    if not hits_raw:
        return [], False, "none"
    exact = hits_raw[0]["similarity"] >= EXACT_THRESHOLD
    return hits_raw, exact, ("exact" if exact else "approx")


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    hits_raw, exact, mode = _retrieve(request)
    header = _header_for(mode)
    llm_answer = generate_answer(request.question, hits_raw, exact=exact)
    answer = header + "\n\n" + llm_answer

    best = hits_raw[0] if hits_raw else None
    return ChatResponse(
        mode=mode,
        exact_match=exact,
        similarity=best["similarity"] if best else 0.0,
        answer=answer,
        matched_question=best["question"] if (best and exact) else None,
        fatwa_link=best["link"] if (best and exact) else None,
        related_fatwas=[FatwaHit(**r) for r in _related(hits_raw)],
    )


def _sse(event: str, data: Dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/chat/stream")
def chat_stream(request: ChatRequest) -> StreamingResponse:
    """
    Server-Sent Events stream:
      - event: meta  -> {mode, exact_match, similarity, header, matched_question, fatwa_link, related_fatwas}
      - event: token -> {t: "<delta text>"}   (many)
      - event: done  -> {}
      - event: error -> {message}
    """

    def event_gen():
        try:
            hits_raw, exact, mode = _retrieve(request)
            best = hits_raw[0] if hits_raw else None
            meta = {
                "mode": mode,
                "exact_match": exact,
                "similarity": best["similarity"] if best else 0.0,
                "header": _header_for(mode),
                "matched_question": best["question"] if (best and exact) else None,
                "fatwa_link": best["link"] if (best and exact) else None,
                "related_fatwas": _related(hits_raw),
            }
            yield _sse("meta", meta)

            for delta in stream_answer(request.question, hits_raw, exact):
                yield _sse("token", {"t": delta})

            yield _sse("done", {})
        except Exception as e:  # surface a clean error to the client
            yield _sse("error", {"message": str(e)})

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # disable proxy buffering (nginx) so tokens flush live
        },
    )
