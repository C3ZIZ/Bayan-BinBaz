import json

from app.gate import GateResult, build_gate_prompt, parse_gate_response, run_gate

HITS = [
    {"id": 11, "question": "حكم استماع الأغاني", "answer": "الأغاني محرمة..."},
    {"id": 22, "question": "حكم المعازف", "answer": "المعازف لا تجوز..."},
]


def _raw(**over):
    payload = {
        "premise_sound": True,
        "premise_issue": "",
        "verdict": "derived",
        "cited_ids": [1],
        "reason": "",
    }
    payload.update(over)
    return json.dumps(payload, ensure_ascii=False)


# ------------------------------------------------------------------- parsing


def test_parses_a_valid_derived_verdict():
    r = parse_gate_response(_raw(), n_candidates=2)
    assert r.verdict == "derived"
    assert r.cited_ids == [1]
    assert r.should_answer


def test_parses_a_valid_direct_verdict():
    assert parse_gate_response(_raw(verdict="direct"), 2).verdict == "direct"


def test_parses_abstain_without_citations():
    r = parse_gate_response(_raw(verdict="abstain", cited_ids=[]), 2)
    assert r.verdict == "abstain"
    assert not r.should_answer


def test_tolerates_code_fences():
    r = parse_gate_response(f"```json\n{_raw()}\n```", 2)
    assert r.verdict == "derived"


def test_tolerates_surrounding_prose():
    r = parse_gate_response(f"هذا حكمي:\n{_raw()}\nانتهى", 2)
    assert r.verdict == "derived"


def test_unparseable_output_fails_closed():
    r = parse_gate_response("ليس هذا JSON على الإطلاق", 2)
    assert r.verdict == "abstain"
    assert r.failed_closed


def test_empty_output_fails_closed():
    assert parse_gate_response("", 2).verdict == "abstain"


def test_unknown_verdict_fails_closed():
    r = parse_gate_response(_raw(verdict="maybe"), 2)
    assert r.verdict == "abstain"
    assert r.failed_closed


# -------------------------------------------------------- safety invariants


def test_false_premise_forces_abstain_even_if_verdict_says_otherwise():
    """The Hajj/Ramadan case: the model must not answer a question whose premise
    it just declared impossible."""
    r = parse_gate_response(
        _raw(verdict="derived", premise_sound=False,
             premise_issue="رمضان الشهر التاسع وأشهر الحج شوال وذو القعدة وذو الحجة",
             cited_ids=[1, 2]),
        n_candidates=2,
    )
    assert r.verdict == "abstain"
    assert r.premise_sound is False
    assert "رمضان" in r.premise_issue
    assert not r.should_answer


def test_answer_without_citations_collapses_to_abstain():
    r = parse_gate_response(_raw(verdict="direct", cited_ids=[]), 2)
    assert r.verdict == "abstain"
    assert not r.should_answer


def test_out_of_range_citation_ids_are_dropped():
    r = parse_gate_response(_raw(cited_ids=[1, 99]), n_candidates=2)
    assert r.cited_ids == [1]


def test_all_citations_out_of_range_collapses_to_abstain():
    r = parse_gate_response(_raw(cited_ids=[99, 100]), n_candidates=2)
    assert r.verdict == "abstain"


def test_zero_and_negative_citation_ids_are_dropped():
    assert parse_gate_response(_raw(cited_ids=[0, -1, 2]), 2).cited_ids == [2]


def test_duplicate_citation_ids_are_collapsed():
    assert parse_gate_response(_raw(cited_ids=[1, 1, 2]), 2).cited_ids == [1, 2]


def test_non_numeric_citation_ids_are_ignored():
    assert parse_gate_response(_raw(cited_ids=["a", 2]), 2).cited_ids == [2]


# ----------------------------------------------------------------- run_gate


def test_run_gate_with_no_hits_abstains_without_calling_the_model():
    called = []

    def chat(**kwargs):
        called.append(kwargs)
        return _raw()

    r = run_gate("سؤال", [], chat=chat)
    assert r.verdict == "abstain"
    assert called == []


def test_run_gate_fails_closed_when_the_api_raises():
    def chat(**kwargs):
        raise RuntimeError("503 service unavailable")

    r = run_gate("سؤال", HITS, chat=chat)
    assert r.verdict == "abstain"
    assert r.failed_closed
    assert "503" in r.reason


def test_run_gate_passes_the_question_and_candidates_through():
    seen = {}

    def chat(**kwargs):
        seen.update(kwargs)
        return _raw()

    run_gate("ما حكم الأغاني؟", HITS, chat=chat)
    assert "ما حكم الأغاني؟" in seen["user"]
    assert "حكم المعازف" in seen["user"]
    assert seen["temperature"] == 0.0


# ------------------------------------------------------------------- prompt


def test_prompt_numbers_candidates_from_one():
    prompt = build_gate_prompt("س", HITS)
    assert "[1]" in prompt and "[2]" in prompt
    assert "[0]" not in prompt


def test_prompt_truncates_long_answers():
    long_hit = [{"id": 1, "question": "س", "answer": "ن " * 5000}]
    assert len(build_gate_prompt("س", long_hit)) < 4000


def test_gate_result_should_answer_requires_citations():
    assert not GateResult(verdict="derived", cited_ids=[]).should_answer
    assert GateResult(verdict="derived", cited_ids=[1]).should_answer


# ------------------------------------- regressions from the code review


def test_string_false_premise_is_not_upgraded_to_true():
    """A gate reply of {"premise_sound": "false"} must abstain. Coercing any
    non-boolean to True made this module fail OPEN — answering exactly the
    impossible question it exists to catch."""
    r = parse_gate_response(
        _raw(verdict="derived", premise_sound="false", cited_ids=[1]), n_candidates=2
    )
    assert r.verdict == "abstain"
    assert r.premise_sound is False


def test_unrecognised_premise_value_fails_closed():
    for bad in ({}, [], "maybe", "غير معروف"):
        r = parse_gate_response(
            _raw(verdict="derived", premise_sound=bad, cited_ids=[1]), 2
        )
        assert r.verdict == "abstain", bad


def test_missing_premise_field_defaults_to_sound():
    payload = json.loads(_raw())
    del payload["premise_sound"]
    r = parse_gate_response(json.dumps(payload, ensure_ascii=False), 2)
    assert r.premise_sound is True
    assert r.verdict == "derived"


def test_numeric_zero_premise_is_false():
    r = parse_gate_response(_raw(premise_sound=0, cited_ids=[1]), 2)
    assert r.verdict == "abstain"


def test_string_true_premise_is_accepted():
    r = parse_gate_response(_raw(premise_sound="true", cited_ids=[1]), 2)
    assert r.premise_sound is True
    assert r.verdict == "derived"


def test_cited_nothing_abstention_is_flagged_as_failed_closed():
    """Otherwise it looks like a considered abstention in the gate metrics."""
    r = parse_gate_response(_raw(verdict="direct", cited_ids=[]), 2)
    assert r.verdict == "abstain"
    assert r.failed_closed is True
