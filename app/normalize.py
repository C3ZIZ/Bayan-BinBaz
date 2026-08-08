"""Arabic text normalization shared by index build and query time.

The SAME function must run on both sides — a mismatch silently degrades
retrieval with no visible error. ``tests/test_index_integrity.py`` enforces that
the index records the ``NORMALIZE_VERSION`` it was built with.

Why this matters here: 86% of answers and 60% of questions in the corpus carry
diacritics, while user queries carry essentially none. Stripping tashkeel also
cuts token count ~10%, which reduces truncation.

Folding is configurable because aggressive normalization can hurt a multilingual
model that saw diacritics during training. ``fold_taa_marbuta`` is off by default
since ة/ه can be meaning-bearing; the golden-set A/B decides whether to enable it.
"""

import re

NORMALIZE_VERSION = "1.0"

# Harakat, tanwin, shadda, sukun, superscript alef, and Quranic annotation marks.
_TASHKEEL = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭ]")
_TATWEEL = re.compile(r"ـ")
_ALEF = re.compile(r"[آأإٱ]")  # آ أ إ ٱ
_ALEF_MAQSURA = re.compile(r"ى")  # ى
_TAA_MARBUTA = re.compile(r"ة")  # ة
_WS = re.compile(r"\s+")

_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")


def normalize_arabic(
    text: str,
    *,
    fold_alef: bool = True,
    fold_yaa: bool = True,
    fold_taa_marbuta: bool = False,
    strip_tashkeel: bool = True,
) -> str:
    """Normalize Arabic text for retrieval. Idempotent; safe on empty input."""
    if not text:
        return ""

    if strip_tashkeel:
        text = _TASHKEEL.sub("", text)
    text = _TATWEEL.sub("", text)
    if fold_alef:
        text = _ALEF.sub("ا", text)
    if fold_yaa:
        text = _ALEF_MAQSURA.sub("ي", text)
    if fold_taa_marbuta:
        text = _TAA_MARBUTA.sub("ه", text)

    text = text.translate(_DIGITS)
    return _WS.sub(" ", text).strip()
