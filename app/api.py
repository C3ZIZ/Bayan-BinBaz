"""Chat endpoints.

Flow: retrieve → gate → generate. The gate decides whether to answer at all;
there is deliberately no similarity threshold anywhere in this module. Measured
on the real index, the false-premise question «إذا جا الحج ورمضان في نفس الوقت»
scores 0.628 while the valid colloquial question «ودي اعرف حكم الاغاني» scores
0.499 — the distributions overlap, so a threshold cannot separate them and the
old EXACT_THRESHOLD has been removed rather than retuned.

Generation is given ONLY the fatwas the gate approved, which is what keeps the
answer's citations honest.
"""

import json
import time
import traceback
from typing import Any, Dict, Iterator, List, Tuple

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from .citations import strip_invalid_markers
from .gate import GateResult, run_gate
from .llm import generate_answer, stream_answer
from .observability import log_query
from .retrieval import get_retriever
from .schemas import ChatRequest, ChatResponse, Citation, FatwaHit

router = APIRouter()

# How many candidates the gate reads. Wider than what the user sees, because
# recall@5 is already 100% but the gate benefits from seeing near-misses to
# judge whether the question is answerable at all.
GATE_CANDIDATES = 8


def _headline(gate: GateResult) -> str:
    if gate.verdict == "direct":
        return "الحالة: وُجدت فتوى للشيخ ابن باز تجيب على هذا السؤال بعينه."
    if gate.verdict == "derived":
        return (
            "الحالة: لا توجد فتوى للشيخ على هذه الحالة بعينها. "
            "ما يلي استنباط من أقرب فتاواه في أصل المسألة، وليس جوابًا مباشرًا منه، "
            "فراجع الفتاوى المشار إليها واسأل أهل العلم لخصوص حالتك."
        )
    if not gate.premise_sound:
        return "الحالة: السؤال مبني على افتراض غير صحيح."
    if gate.failed_closed:
        # Never claim the corpus is empty when we simply could not check it —
        # that is a false statement about the shaykh's fatwas, not a refusal.
        return (
            "الحالة: تعذّر التحقق من الفتاوى حاليًا لعطل مؤقت في الخدمة، "
            "ولم يصدر أي حكم. أعد المحاولة بعد قليل."
        )
    return "الحالة: لا توجد في فتاوى الشيخ ابن باز فتوى تتصل بهذا السؤال."


def _to_hit(h: Dict[str, Any]) -> FatwaHit:
    return FatwaHit(
        id=h["id"],
        question=h["question"],
        title=h["title"],
        link=h["link"],
        similarity=h["similarity"],
        categories=list(h.get("categories") or []),
    )


def _pipeline(request: ChatRequest) -> Tuple[List[Dict], GateResult, List[Dict]]:
    """Retrieve, gate, and resolve the approved sources.

    Returns (candidates, gate result, gate-approved sources in citation order).
    """
    retriever = get_retriever()
    candidates = retriever.search(
        request.question, top_k=max(GATE_CANDIDATES, request.top_k)
    )
    gate = run_gate(request.question, candidates)
    sources = [candidates[n - 1] for n in gate.cited_ids] if gate.should_answer else []
    return candidates, gate, sources


def _citations(text: str, sources: List[Dict[str, Any]]) -> Tuple[str, List[Citation]]:
    """Drop markers the model invented, then resolve the survivors to fatwas."""
    cleaned, used = strip_invalid_markers(text, n_sources=len(sources))
    return cleaned, [
        Citation(
            marker=n,
            id=sources[n - 1]["id"],
            title=sources[n - 1]["title"],
            link=sources[n - 1]["link"],
        )
        for n in used
    ]


def _related(candidates: List[Dict], gate: GateResult, top_k: int) -> List[FatwaHit]:
    """Show related fatwas only when the system actually stands behind them.
    Listing 'closest fatwas' under an abstention invites the user to read a
    ruling the system just declined to give."""
    if gate.verdict == "abstain":
        return []
    return [_to_hit(h) for h in candidates[:top_k]]


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    started = time.monotonic()
    candidates, gate, sources = _pipeline(request)

    if gate.failed_closed:
        # The gate already failed against the same provider; generating would
        # just fail again. Return the service message without a second call.
        return ChatResponse(
            verdict="abstain", answered=False, service_error=True,
            premise_sound=gate.premise_sound, premise_issue=None,
            answer=_headline(gate),
            similarity=candidates[0]["similarity"] if candidates else 0.0,
            citations=[], related_fatwas=[],
        )

    try:
        raw = generate_answer(
            request.question, sources, gate.verdict, gate.premise_issue
        )
    except Exception:
        # The streaming endpoint already handled this; this path did not, and
        # returned a 500 with a stack trace when the LLM provider was out of
        # credits. Surface a clean message and assert no ruling — an upstream
        # failure must never become an ungrounded answer.
        traceback.print_exc()
        raise HTTPException(
            status_code=503,
            detail="تعذّر توليد الجواب حاليًا. حاول مرة أخرى بعد قليل.",
        )

    answer, citations = _citations(raw, sources)

    log_query(
        question=request.question,
        verdict=gate.verdict,
        premise_sound=gate.premise_sound,
        cited_ids=gate.cited_ids,
        top_similarity=candidates[0]["similarity"] if candidates else 0.0,
        latency_ms=(time.monotonic() - started) * 1000,
        gate_failed_closed=gate.failed_closed,
    )

    return ChatResponse(
        verdict=gate.verdict,
        answered=gate.should_answer,
        service_error=False,
        premise_sound=gate.premise_sound,
        premise_issue=gate.premise_issue or None,
        answer=_headline(gate) + "\n\n" + answer,
        similarity=candidates[0]["similarity"] if candidates else 0.0,
        citations=citations,
        related_fatwas=_related(candidates, gate, request.top_k),
    )


def _sse(event: str, data: Dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/chat/stream")
def chat_stream(request: ChatRequest) -> StreamingResponse:
    """Server-Sent Events stream.

      meta  -> {verdict, answered, premise_sound, premise_issue, header,
                similarity, sources, related_fatwas}
      token -> {t: "<delta>"}   (many)
      done  -> {citations: [...]}   resolved after the full text is known
      error -> {message}

    The gate runs before the first token, so the client never sees a claim that
    a later check would have to retract.
    """

    started = time.monotonic()

    def event_gen() -> Iterator[str]:
        try:
            candidates, gate, sources = _pipeline(request)

            yield _sse("meta", {
                "verdict": gate.verdict,
                "answered": gate.should_answer,
                "service_error": gate.failed_closed,
                "premise_sound": gate.premise_sound,
                "premise_issue": gate.premise_issue or None,
                "header": _headline(gate),
                "similarity": candidates[0]["similarity"] if candidates else 0.0,
                "sources": [
                    {"marker": i, "id": s["id"], "title": s["title"], "link": s["link"]}
                    for i, s in enumerate(sources, start=1)
                ],
                "related_fatwas": [
                    h.model_dump() for h in _related(candidates, gate, request.top_k)
                ],
            })

            if gate.failed_closed:
                # Same reasoning as the non-streaming path: do not issue a
                # second call to a provider that just failed.
                yield _sse("done", {"citations": []})
                return

            parts: List[str] = []
            for delta in stream_answer(
                request.question, sources, gate.verdict, gate.premise_issue
            ):
                parts.append(delta)
                yield _sse("token", {"t": delta})

            _, citations = _citations("".join(parts), sources)
            yield _sse("done", {"citations": [c.model_dump() for c in citations]})

            log_query(
                question=request.question,
                verdict=gate.verdict,
                premise_sound=gate.premise_sound,
                cited_ids=gate.cited_ids,
                top_similarity=candidates[0]["similarity"] if candidates else 0.0,
                latency_ms=(time.monotonic() - started) * 1000,
                gate_failed_closed=gate.failed_closed,
                extra={"streamed": True},
            )

        except Exception:
            # Never stream the exception text: it can carry server paths, env
            # var names, and upstream provider response bodies to an
            # unauthenticated client. Detail goes to the server log only.
            traceback.print_exc()
            yield _sse("error", {"message": "تعذّر توليد الجواب. حاول مرة أخرى."})

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # disable proxy buffering so tokens flush live
        },
    )
