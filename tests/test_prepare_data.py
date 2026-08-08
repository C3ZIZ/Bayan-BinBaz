import numpy as np
import pandas as pd

from prepare_data import normalize_fatwa_table, parse_categories


def test_parse_categories_from_stringified_list():
    assert parse_categories("['التفسير']") == ["التفسير"]


def test_parse_categories_handles_multiple():
    assert parse_categories("['أ', 'ب']") == ["أ", "ب"]


def test_parse_categories_handles_garbage():
    assert parse_categories("not a list") == []
    assert parse_categories(None) == []
    assert parse_categories(float("nan")) == []


def test_parse_categories_handles_empty_list_string():
    assert parse_categories("[]") == []


def test_parse_categories_passes_through_real_list():
    assert parse_categories(["أ"]) == ["أ"]


def _frame(**over):
    base = {
        "questions": ["س1", "س2"],
        "answers": ["ج1", "ج2"],
        "titles": ["ع1", "ع2"],
        "links": ["l1", "l2"],
        "categories": ["['أ']", "['ب']"],
    }
    base.update(over)
    return pd.DataFrame(base)


def test_drops_rows_with_null_answer_instead_of_stringifying():
    df = normalize_fatwa_table(_frame(answers=["ج1", np.nan]))
    assert len(df) == 1
    assert "nan" not in df["answer"].tolist()


def test_drops_rows_with_null_question():
    df = normalize_fatwa_table(_frame(questions=["س1", None]))
    assert len(df) == 1


def test_drops_literal_nan_strings():
    df = normalize_fatwa_table(_frame(answers=["ج1", "nan"]))
    assert len(df) == 1


def test_drops_whitespace_only_answers():
    df = normalize_fatwa_table(_frame(answers=["ج1", "   "]))
    assert len(df) == 1


def test_categories_become_real_lists():
    df = normalize_fatwa_table(_frame())
    assert df["categories"].iloc[0] == ["أ"]
    assert isinstance(df["categories"].iloc[0], list)


def test_renames_plural_columns():
    df = normalize_fatwa_table(_frame())
    assert {"question", "answer", "title", "link", "categories"}.issubset(df.columns)


def test_missing_optional_columns_are_created():
    df = normalize_fatwa_table(pd.DataFrame({"questions": ["س"], "answers": ["ج"]}))
    assert df["title"].iloc[0] == ""
    assert df["categories"].iloc[0] == []
    assert df["link"].iloc[0] == ""


def test_strips_surrounding_whitespace():
    df = normalize_fatwa_table(_frame(questions=["  س1  ", "س2"]))
    assert df["question"].iloc[0] == "س1"


def test_output_column_order_is_stable():
    df = normalize_fatwa_table(_frame())
    assert list(df.columns) == ["question", "answer", "title", "link", "categories"]
