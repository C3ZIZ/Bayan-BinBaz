import json

from app import observability


def _log(monkeypatch, path, **over):
    monkeypatch.setattr(observability, "QUERY_LOG_PATH", str(path))
    kwargs = dict(
        question="ما حكم كذا؟", verdict="derived", premise_sound=True,
        cited_ids=[1, 2], top_similarity=0.61, latency_ms=1234.5,
    )
    kwargs.update(over)
    observability.log_query(**kwargs)


def test_writes_a_jsonl_record(tmp_path, monkeypatch):
    log = tmp_path / "q.jsonl"
    _log(monkeypatch, log)
    record = json.loads(log.read_text(encoding="utf-8").strip())
    assert record["verdict"] == "derived"
    assert record["n_citations"] == 2
    assert record["top_similarity"] == 0.61


def test_appends_rather_than_overwrites(tmp_path, monkeypatch):
    log = tmp_path / "q.jsonl"
    _log(monkeypatch, log)
    _log(monkeypatch, log, verdict="abstain")
    lines = log.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[1])["verdict"] == "abstain"


def test_disabled_by_default(tmp_path, monkeypatch):
    monkeypatch.setattr(observability, "QUERY_LOG_PATH", "")
    observability.log_query("س", "derived", True, [1], 0.5, 10.0)
    assert list(tmp_path.iterdir()) == []


def test_does_not_record_client_identity(tmp_path, monkeypatch):
    log = tmp_path / "q.jsonl"
    _log(monkeypatch, log)
    record = json.loads(log.read_text(encoding="utf-8").strip())
    for forbidden in ("ip", "client", "user", "session", "headers"):
        assert forbidden not in record


def test_creates_parent_directories(tmp_path, monkeypatch):
    log = tmp_path / "nested" / "deep" / "q.jsonl"
    _log(monkeypatch, log)
    assert log.exists()


def test_never_raises_on_an_unwritable_path(monkeypatch):
    monkeypatch.setattr(observability, "QUERY_LOG_PATH", "/proc/definitely/not/writable")
    observability.log_query("س", "derived", True, [], 0.1, 5.0)  # must not raise


def test_extra_fields_are_merged(tmp_path, monkeypatch):
    log = tmp_path / "q.jsonl"
    _log(monkeypatch, log, extra={"kind": "smoke"})
    assert json.loads(log.read_text(encoding="utf-8").strip())["kind"] == "smoke"
