"""The answerability gate.

This is the component that fixes fabricated rulings. Measurement showed no
cosine threshold can separate answerable from unanswerable questions on this
corpus: the false-premise question «إذا جا الحج ورمضان في نفس الوقت» scores 0.628
against the index, *higher* than the perfectly valid colloquial question
«ودي اعرف حكم الاغاني» at 0.499. The distributions overlap, so the decision has
to be made by something that reads the candidates rather than ranks them.

The gate is a single structured LLM call placed between retrieval and
generation. It returns three verdicts:

  direct  — a retrieved fatwa answers the question itself
  derived — no fatwa is exact, but the retrieved ones establish the ruling.
            THIS IS THE COMMON CASE: a fatwa's own question self-matches at a
            median cosine of only 0.815, so demanding an exact match would
            refuse most legitimate questions.
  abstain — the fatwas do not establish an answer, the premise is false, or the
            question is out of scope.

It fails CLOSED. On API error, timeout, or unparseable output the verdict is
``abstain`` — never a silent fallback to answering ungrounded, which would
reinstate the original bug precisely when the system is under stress.
"""

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from .textutil import truncate_at_sentence

GATE_MODEL = os.getenv("GATE_MODEL", os.getenv("LLM_API_MODEL", "Qwen/Qwen2.5-72B-Instruct"))
GATE_MAX_TOKENS = int(os.getenv("GATE_MAX_TOKENS", "400"))
GATE_TEMPERATURE = float(os.getenv("GATE_TEMPERATURE", "0.0"))
GATE_SNIPPET_CHARS = int(os.getenv("GATE_SNIPPET_CHARS", "600"))

VERDICTS = ("direct", "derived", "abstain")

GATE_SYSTEM_PROMPT = """
أنت مدقق تحكيمي في نظام يجيب عن الأسئلة الشرعية اعتمادًا على فتاوى الشيخ عبد العزيز بن باز رحمه الله فقط.
مهمتك ليست الإجابة، بل الحكم على إمكانية الإجابة من الفتاوى المعروضة عليك.

افحص أولًا صحة افتراض السؤال:
- بعض الأسئلة تقوم على افتراض مستحيل (مثل: اجتماع رمضان والحج في وقت واحد، ورمضان هو الشهر التاسع وأشهر الحج شوال وذو القعدة وذو الحجة).
- إذا كان الافتراض مستحيلًا فاجعل premise_sound = false، وبيّن الخلل في premise_issue.

ثم أصدر الحكم:
- "direct": توجد فتوى تجيب عن السؤال نفسه.
- "derived": لا توجد فتوى بعينها، لكن الفتاوى المعروضة تُثبت الحكم أو القاعدة التي يُبنى عليها الجواب. هذه هي الحالة الغالبة، فلا تتشدد.
- "abstain": الفتاوى لا تُثبت جوابًا، أو الافتراض باطل، أو السؤال خارج نطاق الفتاوى.

قواعد ملزمة:
- "abstain" إذا كان premise_sound = false.
- ضع في cited_ids أرقام الفتاوى المفيدة فعلًا فقط، ولا تضع رقمًا لا يفيد.
- لا تختر "abstain" لمجرد عدم وجود تطابق حرفي؛ إن كانت الفتاوى تُثبت الحكم فاختر "derived".
- أعد كائن JSON فقط بلا أي نص آخر.

صيغة الإخراج:
{"premise_sound": true|false, "premise_issue": "", "verdict": "direct|derived|abstain", "cited_ids": [1,2], "reason": ""}
""".strip()


@dataclass
class GateResult:
    verdict: str
    premise_sound: bool = True
    premise_issue: str = ""
    cited_ids: List[int] = field(default_factory=list)
    reason: str = ""
    failed_closed: bool = False

    @property
    def should_answer(self) -> bool:
        return self.verdict in ("direct", "derived") and bool(self.cited_ids)


def build_gate_prompt(question: str, hits: List[Dict[str, Any]]) -> str:
    blocks = []
    for i, hit in enumerate(hits, start=1):
        blocks.append(
            f"[{i}]\nالسؤال: {(hit.get('question') or '').strip()}\n"
            f"الجواب: {truncate_at_sentence(hit.get('answer') or '', GATE_SNIPPET_CHARS)}"
        )
    return (
        f"سؤال المستخدم:\n{question}\n\n"
        f"الفتاوى المرشحة:\n\n" + "\n\n".join(blocks) + "\n\nأصدر حكمك بصيغة JSON فقط."
    )


_TRUTHY = {"true", "yes", "1"}
_FALSY = {"false", "no", "0"}


def _coerce_premise_sound(value: Any) -> bool:
    """Interpret the gate's ``premise_sound`` field, failing CLOSED.

    A missing field means the gate saw nothing wrong, so the premise is sound.
    But a *present* value that is not recognisably true — the string "false",
    0, None, an object — must not be silently upgraded to True: that is exactly
    how the impossible Hajj/Ramadan question would get answered.
    """
    if value is None:
        return True
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in _TRUTHY:
            return True
        if text in _FALSY:
            return False
    return False


def _extract_json(raw: str) -> Optional[Dict[str, Any]]:
    """Pull the first JSON object out of a model response, tolerating code
    fences and surrounding prose."""
    if not raw:
        return None
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end <= start:
            return None
        candidate = raw[start : end + 1]
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def parse_gate_response(raw: str, n_candidates: int) -> GateResult:
    """Validate a raw gate response. Anything malformed or self-inconsistent
    collapses to abstain."""
    data = _extract_json(raw)
    if data is None:
        return GateResult(verdict="abstain", reason="unparseable gate response",
                          failed_closed=True)

    verdict = str(data.get("verdict", "")).strip().lower()
    if verdict not in VERDICTS:
        return GateResult(verdict="abstain", reason=f"unknown verdict {verdict!r}",
                          failed_closed=True)

    premise_sound = _coerce_premise_sound(data.get("premise_sound"))
    premise_issue = str(data.get("premise_issue") or "").strip()

    # Keep only ids the gate was actually shown; a hallucinated source is worse
    # than no source.
    cited: List[int] = []
    for value in data.get("cited_ids") or []:
        try:
            n = int(value)
        except (TypeError, ValueError):
            continue
        if 1 <= n <= n_candidates and n not in cited:
            cited.append(n)

    if not premise_sound:
        return GateResult(
            verdict="abstain", premise_sound=False, premise_issue=premise_issue,
            reason=str(data.get("reason") or "false premise"),
        )

    if verdict in ("direct", "derived") and not cited:
        # Flagged as a closed failure so it shows up in the gate metrics rather
        # than looking like a considered abstention.
        return GateResult(
            verdict="abstain",
            reason="verdict claimed an answer but cited no usable fatwa",
            failed_closed=True,
        )

    return GateResult(
        verdict=verdict, premise_sound=True, premise_issue=premise_issue,
        cited_ids=cited, reason=str(data.get("reason") or ""),
    )


def run_gate(
    question: str,
    hits: List[Dict[str, Any]],
    chat: Optional[Callable[..., str]] = None,
) -> GateResult:
    """Run the gate. ``chat`` is injectable so tests need no network."""
    if not hits:
        return GateResult(verdict="abstain", reason="no candidates retrieved")

    if chat is None:
        from .llm import chat_completion as chat

    try:
        raw = chat(
            system=GATE_SYSTEM_PROMPT,
            user=build_gate_prompt(question, hits),
            model=GATE_MODEL,
            max_tokens=GATE_MAX_TOKENS,
            temperature=GATE_TEMPERATURE,
        )
    except Exception as exc:  # network, auth, rate limit, timeout
        return GateResult(verdict="abstain", reason=f"gate call failed: {exc}",
                          failed_closed=True)

    return parse_gate_response(raw, n_candidates=len(hits))
