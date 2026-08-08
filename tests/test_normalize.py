from app.normalize import NORMALIZE_VERSION, normalize_arabic


def test_strips_tashkeel():
    assert normalize_arabic("الصَّلَاةُ") == "الصلاة"


def test_strips_tatweel():
    assert normalize_arabic("الصــلاة") == "الصلاة"


def test_folds_alef_variants():
    assert normalize_arabic("أإآا") == "اااا"


def test_folds_alef_maqsura_to_yaa():
    assert normalize_arabic("على") == "علي"


def test_taa_marbuta_preserved_by_default():
    assert normalize_arabic("صلاة") == "صلاة"


def test_taa_marbuta_folded_when_requested():
    assert normalize_arabic("صلاة", fold_taa_marbuta=True) == "صلاه"


def test_converts_arabic_indic_digits():
    assert normalize_arabic("١٢٣") == "123"


def test_collapses_whitespace():
    assert normalize_arabic("  الصلاة   واجبة \n\n علينا ") == "الصلاة واجبة علينا"


def test_is_idempotent():
    once = normalize_arabic("الصَّلَاةُ   واجِبَة")
    assert normalize_arabic(once) == once


def test_handles_empty_and_whitespace():
    assert normalize_arabic("") == ""
    assert normalize_arabic("   ") == ""


def test_leaves_latin_untouched():
    assert normalize_arabic("BGE-M3 v2") == "BGE-M3 v2"


def test_version_is_a_nonempty_string():
    assert isinstance(NORMALIZE_VERSION, str) and NORMALIZE_VERSION


def test_disabling_tashkeel_strip_preserves_diacritics():
    assert normalize_arabic("الصَّلَاةُ", strip_tashkeel=False) != "الصلاة"


def test_real_corpus_sample():
    # Taken verbatim from data/processed/fatwas.parquet (id 960 title).
    assert normalize_arabic("حكم التنقل بين مساجد التراويح للصوت النَّدي") == (
        "حكم التنقل بين مساجد التراويح للصوت الندي"
    )
