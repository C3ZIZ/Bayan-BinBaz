import json
from pathlib import Path

import pandas as pd
import pytest

GOLDEN = Path("eval/datasets/golden.jsonl")
DERIVED = Path("eval/datasets/derived.jsonl")
ADVERSARIAL = Path("eval/datasets/adversarial.jsonl")
META = Path("data/index/fatwas_meta.parquet")


def _load(p: Path):
    return [
        json.loads(line)
        for line in p.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _valid_ids():
    return set(pd.read_parquet(META)["id"].tolist())


# --------------------------------------------------------------------- golden

golden_only = pytest.mark.skipif(not GOLDEN.exists(), reason="golden set not built yet")


@pytest.fixture(scope="module")
def golden():
    return _load(GOLDEN)


@golden_only
def test_golden_has_expected_size(golden):
    assert len(golden) == 200


@golden_only
def test_golden_rows_have_required_fields(golden):
    for r in golden:
        assert set(r) >= {"query", "relevant_id", "kind"}
        assert isinstance(r["relevant_id"], int)
        assert r["query"].strip()


@golden_only
def test_golden_kinds_are_balanced(golden):
    kinds = [r["kind"] for r in golden]
    assert kinds.count("verbatim") == 100
    assert kinds.count("paraphrase") == 100


@golden_only
@pytest.mark.skipif(not META.exists(), reason="index meta not built yet")
def test_golden_relevant_ids_exist(golden):
    assert {r["relevant_id"] for r in golden} <= _valid_ids()


@golden_only
def test_golden_has_no_duplicate_queries(golden):
    queries = [r["query"] for r in golden]
    assert len(queries) == len(set(queries))


# -------------------------------------------------------------------- derived

derived_only = pytest.mark.skipif(
    not DERIVED.exists(), reason="derived set not built yet"
)


@pytest.fixture(scope="module")
def derived():
    return _load(DERIVED)


@derived_only
def test_derived_set_shape(derived):
    assert len(derived) == 60
    for r in derived:
        assert r["expected_verdict"] == "derived"
        assert r["query"].strip()
        assert isinstance(r["supporting_ids"], list) and r["supporting_ids"]
        assert r["note"].strip()


@derived_only
def test_derived_kinds_are_balanced(derived):
    kinds = [r["kind"] for r in derived]
    for kind in ("dialect", "narrower", "application"):
        assert kinds.count(kind) == 20, f"{kind}: {kinds.count(kind)}"


@derived_only
def test_derived_sources_are_real_urls(derived):
    for r in derived:
        assert r["source"].startswith("http"), r["query"]


@derived_only
@pytest.mark.skipif(not META.exists(), reason="index meta not built yet")
def test_derived_supporting_ids_exist(derived):
    valid = _valid_ids()
    for r in derived:
        assert set(r["supporting_ids"]) <= valid, r["query"]


@derived_only
def test_derived_queries_are_not_verbatim_corpus_questions(derived):
    """The whole point of this set: if the query already exists verbatim in the
    corpus it tests exact match, not derivation."""
    if not META.exists():
        pytest.skip("index meta not built yet")
    corpus = set(pd.read_parquet(META)["question"].astype(str).str.strip())
    for r in derived:
        assert r["query"].strip() not in corpus, r["query"]


# ---------------------------------------------------------------- adversarial

adversarial_only = pytest.mark.skipif(
    not ADVERSARIAL.exists(), reason="adversarial set not built yet"
)


@pytest.fixture(scope="module")
def adversarial():
    return _load(ADVERSARIAL)


@adversarial_only
def test_adversarial_set_shape(adversarial):
    assert len(adversarial) == 60
    for r in adversarial:
        assert r["expected_verdict"] == "abstain"
        assert r["query"].strip()


@adversarial_only
def test_adversarial_reasons_are_balanced(adversarial):
    reasons = [r["reason"] for r in adversarial]
    for reason in ("false_premise", "out_of_scope", "gibberish"):
        assert reasons.count(reason) == 20, f"{reason}: {reasons.count(reason)}"


@adversarial_only
def test_false_premise_rows_explain_the_defect(adversarial):
    for r in adversarial:
        if r["reason"] == "false_premise":
            assert r["note"].strip(), r["query"]


@adversarial_only
def test_the_reported_hajj_ramadan_case_is_covered(adversarial):
    """Regression guard for the bug that motivated this work."""
    rows = [r for r in adversarial if "الحج" in r["query"] and "رمضان" in r["query"]]
    assert rows, "the reported Hajj/Ramadan question is missing"
    assert all(r["expected_verdict"] == "abstain" for r in rows)


@adversarial_only
def test_adversarial_has_no_duplicate_queries(adversarial):
    queries = [r["query"] for r in adversarial]
    assert len(queries) == len(set(queries))


# ------------------------------------------------------------------ crossover


@derived_only
@adversarial_only
def test_derived_and_adversarial_do_not_overlap(derived, adversarial):
    """A query cannot be both answerable and required to abstain."""
    assert not ({r["query"] for r in derived} & {r["query"] for r in adversarial})
