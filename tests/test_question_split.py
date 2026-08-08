from app.question_split import is_multipart, split_questions


def test_single_question_is_returned_unchanged():
    q = "ما حكم صيام يوم عرفة لغير الحاج؟"
    assert split_questions(q) == [q]


def test_single_question_without_a_question_mark():
    q = "ودي اعرف حكم الاغاني وش رايك فيها"
    assert split_questions(q) == [q]


def test_the_reported_three_part_question_splits():
    """The exact question that retrieved neither pork nor Maghrib rakʿahs."""
    parts = split_questions("حكم اكل الخنزير، كمان كم ركعات المغرب؟ وهل اقدر اصلي العشاء ٥ ركعات؟")
    assert len(parts) == 3
    assert "الخنزير" in parts[0]
    assert "المغرب" in parts[1]
    assert "العشاء" in parts[2]


def test_splits_on_question_marks():
    parts = split_questions("ما حكم الأغاني؟ وما حكم المعازف؟")
    assert len(parts) == 2
    assert "الأغاني" in parts[0] and "المعازف" in parts[1]


def test_splits_on_connector_before_an_interrogative():
    parts = split_questions("حكم اكل الخنزير وهل يجوز بيعه")
    assert len(parts) == 2


def test_does_not_split_on_waw_inside_one_question():
    """«و» joining nouns must not be treated as a question boundary."""
    q = "ما حكم بيع وشراء الذهب بالتقسيط"
    assert split_questions(q) == [q]


def test_does_not_split_when_waw_is_followed_by_a_non_interrogative():
    q = "ما حكم الصلاة وهو جالس على الكرسي"
    assert split_questions(q) == [q]


def test_drops_fragments_that_are_too_short():
    parts = split_questions("ما حكم الأغاني؟ نعم؟")
    assert all(len(p.split()) >= 2 for p in parts)


def test_caps_the_number_of_parts():
    parts = split_questions("ما حكم أ؟ وما حكم ب؟ وما حكم ج؟ وما حكم د؟ وما حكم هـ؟ وما حكم و؟")
    assert len(parts) <= 4


def test_empty_input_is_safe():
    assert split_questions("") == []
    assert split_questions("   ") == []


def test_is_multipart_flags_correctly():
    assert is_multipart("ما حكم الأغاني؟ وما حكم المعازف؟") is True
    assert is_multipart("ما حكم الأغاني؟") is False


def test_latin_question_mark_also_splits():
    parts = split_questions("ما حكم الأغاني? وما حكم المعازف?")
    assert len(parts) == 2
