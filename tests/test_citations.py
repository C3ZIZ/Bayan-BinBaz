import pytest

from app.citations import extract_markers, strip_invalid_markers


def test_extract_single_marker():
    assert extract_markers("الحكم كذا [1].") == [1]


def test_extract_multiple_markers_in_order():
    assert extract_markers("أولًا [1] وثانيًا [3] وثالثًا [2]") == [1, 3, 2]


def test_extract_deduplicates_but_keeps_first_order():
    assert extract_markers("[2] ثم [1] ثم [2]") == [2, 1]


def test_extract_handles_adjacent_markers():
    assert extract_markers("الحكم [1][2] كذا") == [1, 2]


def test_extract_ignores_non_numeric_brackets():
    assert extract_markers("قال [الشيخ] كذا [1]") == [1]


def test_extract_on_text_without_markers():
    assert extract_markers("لا توجد إحالات هنا") == []


def test_extract_on_empty_text():
    assert extract_markers("") == []


def test_strip_removes_out_of_range_markers():
    text, used = strip_invalid_markers("الحكم [1] وكذلك [7].", n_sources=3)
    assert "[7]" not in text
    assert used == [1]


def test_strip_keeps_valid_markers():
    text, used = strip_invalid_markers("الحكم [1] و [2].", n_sources=2)
    assert "[1]" in text and "[2]" in text
    assert used == [1, 2]


def test_strip_rejects_zero_and_negative_markers():
    text, used = strip_invalid_markers("الحكم [0] كذا", n_sources=3)
    assert "[0]" not in text
    assert used == []


def test_strip_with_no_sources_removes_everything():
    text, used = strip_invalid_markers("الحكم [1] كذا", n_sources=0)
    assert "[1]" not in text
    assert used == []


def test_strip_does_not_leave_double_spaces():
    text, _ = strip_invalid_markers("الحكم [9] كذا", n_sources=1)
    assert "  " not in text


def test_strip_rejects_negative_n_sources():
    with pytest.raises(ValueError):
        strip_invalid_markers("x", n_sources=-1)


def test_strip_preserves_text_without_markers():
    text, used = strip_invalid_markers("نص بلا إحالات", n_sources=2)
    assert text == "نص بلا إحالات"
    assert used == []
