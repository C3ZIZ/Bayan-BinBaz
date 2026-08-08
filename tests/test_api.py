"""End-to-end endpoint behaviour with retrieval and the LLM stubbed out.

The point of these tests is the contract that fixes the reported bug: when the
gate abstains, the response must contain no ruling, no citations, and no list of
"closest fatwas" that would invite the user to read a ruling into the refusal.
"""

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import api as api_module
from app.gate import GateResult

CANDIDATES = [
    {
        "id": 101, "question": "حكم استماع الأغاني", "title": "حكم الأغاني",
        "answer": "الأغاني محرمة لما فيها من صد عن ذكر الله.",
        "link": "https://binbaz.org.sa/fatwas/101", "categories": ["الشعر والأغاني"],
        "similarity": 0.62,
    },
    {
        "id": 102, "question": "حكم المعازف", "title": "حكم المعازف",
        "answer": "المعازف لا تجوز.", "link": "https://binbaz.org.sa/fatwas/102",
        "categories": ["الشعر والأغاني"], "similarity": 0.58,
    },
]


class FakeRetriever:
    def search(self, question, top_k=5):
        return [dict(c) for c in CANDIDATES[:top_k]]


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(api_module, "get_retriever", lambda: FakeRetriever())
    app = FastAPI()
    app.include_router(api_module.router, prefix="/api")
    return TestClient(app)


def _stub_llm(monkeypatch, text):
    monkeypatch.setattr(api_module, "generate_answer",
                        lambda q, s, v, p="": text)
    monkeypatch.setattr(api_module, "stream_answer",
                        lambda q, s, v, p="": iter(text.split(" ")))


def _stub_gate(monkeypatch, result):
    monkeypatch.setattr(api_module, "run_gate", lambda q, c: result)


# ------------------------------------------------------------------ answering


def test_derived_answer_resolves_citations(client, monkeypatch):
    _stub_gate(monkeypatch, GateResult(verdict="derived", cited_ids=[1, 2]))
    _stub_llm(monkeypatch, "قرر الشيخ التحريم [1] وكذلك المعازف [2].")

    body = client.post("/api/chat", json={"question": "حكم الاغاني"}).json()
    assert body["verdict"] == "derived"
    assert body["answered"] is True
    assert [c["marker"] for c in body["citations"]] == [1, 2]
    assert body["citations"][0]["id"] == 101
    assert body["citations"][0]["link"].endswith("/101")


def test_direct_answer_reports_direct_verdict(client, monkeypatch):
    _stub_gate(monkeypatch, GateResult(verdict="direct", cited_ids=[1]))
    _stub_llm(monkeypatch, "الحكم التحريم [1].")
    assert client.post("/api/chat", json={"question": "س"}).json()["verdict"] == "direct"


def test_invented_citation_markers_are_removed(client, monkeypatch):
    """A marker pointing at a source the gate never approved must not reach the
    user — an unverifiable citation is worse than none."""
    _stub_gate(monkeypatch, GateResult(verdict="derived", cited_ids=[1]))
    _stub_llm(monkeypatch, "حكم كذا [1] وأيضًا [7] و [9].")

    body = client.post("/api/chat", json={"question": "س"}).json()
    assert "[7]" not in body["answer"] and "[9]" not in body["answer"]
    assert [c["marker"] for c in body["citations"]] == [1]


# ----------------------------------------------------------------- abstention


def test_false_premise_returns_no_ruling_and_no_citations(client, monkeypatch):
    """The reported bug: Hajj and Ramadan cannot coincide."""
    _stub_gate(monkeypatch, GateResult(
        verdict="abstain", premise_sound=False,
        premise_issue="رمضان الشهر التاسع وأشهر الحج شوال وذو القعدة وذو الحجة",
    ))
    _stub_llm(monkeypatch, "السؤال مبني على افتراض لا يقع.")

    body = client.post(
        "/api/chat",
        json={"question": "اذا جا الحج ورمضان في نفس الوقت نصوم ولا نحج؟"},
    ).json()

    assert body["verdict"] == "abstain"
    assert body["answered"] is False
    assert body["premise_sound"] is False
    assert body["citations"] == []
    assert "رمضان" in body["premise_issue"]


def test_abstention_does_not_list_related_fatwas(client, monkeypatch):
    """Listing 'closest fatwas' under a refusal invites the user to read a
    ruling the system just declined to give."""
    _stub_gate(monkeypatch, GateResult(verdict="abstain"))
    _stub_llm(monkeypatch, "لا توجد فتوى.")

    body = client.post("/api/chat", json={"question": "كيف أطبخ الكبسة؟"}).json()
    assert body["related_fatwas"] == []


def test_answering_verdicts_do_list_related_fatwas(client, monkeypatch):
    _stub_gate(monkeypatch, GateResult(verdict="derived", cited_ids=[1]))
    _stub_llm(monkeypatch, "حكم كذا [1].")
    body = client.post("/api/chat", json={"question": "س"}).json()
    assert len(body["related_fatwas"]) == 2


def test_gate_failure_abstains_rather_than_answering(client, monkeypatch):
    _stub_gate(monkeypatch, GateResult(
        verdict="abstain", reason="gate call failed", failed_closed=True))
    _stub_llm(monkeypatch, "لا توجد فتوى.")

    body = client.post("/api/chat", json={"question": "س"}).json()
    assert body["answered"] is False
    assert body["citations"] == []


# ------------------------------------------------------------------ validation


def test_empty_question_is_rejected(client):
    assert client.post("/api/chat", json={"question": ""}).status_code == 422


def test_oversized_top_k_is_rejected(client):
    assert client.post(
        "/api/chat", json={"question": "س", "top_k": 100000}
    ).status_code == 422


def test_zero_top_k_is_rejected(client):
    assert client.post("/api/chat", json={"question": "س", "top_k": 0}).status_code == 422


# --------------------------------------------------------------------- stream


def _events(raw: str):
    out = []
    for frame in raw.split("\n\n"):
        if not frame.strip():
            continue
        name, payload = None, []
        for line in frame.split("\n"):
            if line.startswith("event:"):
                name = line[6:].strip()
            elif line.startswith("data:"):
                payload.append(line[5:].lstrip())
        out.append((name, json.loads("\n".join(payload))))
    return out


def test_stream_emits_meta_tokens_and_done(client, monkeypatch):
    _stub_gate(monkeypatch, GateResult(verdict="derived", cited_ids=[1]))
    _stub_llm(monkeypatch, "حكم كذا [1].")

    events = _events(client.post("/api/chat/stream", json={"question": "س"}).text)
    names = [n for n, _ in events]
    assert names[0] == "meta"
    assert "token" in names
    assert names[-1] == "done"


def test_stream_meta_carries_the_verdict_before_any_token(client, monkeypatch):
    _stub_gate(monkeypatch, GateResult(verdict="abstain", premise_sound=False,
                                       premise_issue="افتراض مستحيل"))
    _stub_llm(monkeypatch, "لا يقع هذا.")

    events = _events(client.post("/api/chat/stream", json={"question": "س"}).text)
    name, meta = events[0]
    assert name == "meta"
    assert meta["verdict"] == "abstain"
    assert meta["premise_sound"] is False
    assert meta["related_fatwas"] == []


def test_stream_done_event_carries_resolved_citations(client, monkeypatch):
    _stub_gate(monkeypatch, GateResult(verdict="derived", cited_ids=[1]))
    _stub_llm(monkeypatch, "حكم [1].")

    events = _events(client.post("/api/chat/stream", json={"question": "س"}).text)
    _, done = events[-1]
    assert done["citations"][0]["id"] == 101
