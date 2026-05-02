from pydantic import ValidationError

from app.models import GeneratedCard


def test_generated_card_accepts_valid_noun_payload() -> None:
    payload = {
        "dutch_word": "school",
        "russian_translation": "школа",
        "part_of_speech": "noun",
        "ipa_transcription": "sxoːl",
        "example_sentence_nl": "Mijn school is dichtbij.",
        "example_sentence_ru": "Моя школа находится рядом.",
        "lesson_topic": "De school",
        "tags": ["school", "lesson-3"],
        "article": "de",
        "plural_form": "scholen",
        "front_hint": "школа (множественное число?)",
        "verb_forms": None,
        "adjective_forms": None,
    }

    card = GeneratedCard.model_validate(payload)
    assert card.plural_form == "scholen"


def test_generated_card_rejects_noun_without_plural() -> None:
    payload = {
        "dutch_word": "school",
        "russian_translation": "школа",
        "part_of_speech": "noun",
        "ipa_transcription": "sxoːl",
        "example_sentence_nl": "Mijn school is dichtbij.",
        "example_sentence_ru": "Моя школа находится рядом.",
        "lesson_topic": "De school",
        "tags": ["school"],
        "article": "de",
        "front_hint": "школа (множественное число?)",
        "verb_forms": None,
        "adjective_forms": None,
    }

    try:
        GeneratedCard.model_validate(payload)
    except ValidationError as exc:
        assert "plural_form" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("validation should have failed")


def test_generated_card_accepts_regular_adjective_without_adjective_forms() -> None:
    payload = {
        "dutch_word": "mooi",
        "russian_translation": "красивый",
        "part_of_speech": "adjective",
        "ipa_transcription": "moːi",
        "example_sentence_nl": "Het is een mooi huis.",
        "example_sentence_ru": "Это красивый дом.",
        "lesson_topic": "Het huis",
        "tags": ["house", "adjective"],
        "article": None,
        "plural_form": None,
        "front_hint": None,
        "verb_forms": None,
        "adjective_forms": None,
    }

    card = GeneratedCard.model_validate(payload)

    assert card.adjective_forms is None


def test_generated_card_accepts_indeclinable_adjective_example() -> None:
    payload = {
        "dutch_word": "gouden",
        "russian_translation": "золотой",
        "part_of_speech": "adjective",
        "ipa_transcription": "ˈɣʌudə(n)",
        "example_sentence_nl": "Zij draagt een gouden ring.",
        "example_sentence_ru": "Она носит золотое кольцо.",
        "lesson_topic": "Kleding",
        "tags": ["clothing", "adjective"],
        "article": None,
        "plural_form": None,
        "front_hint": None,
        "verb_forms": None,
        "adjective_forms": {
            "onverbuigbaar_example": "gouden ring",
            "learner_note": "Stofadjectief op -en.",
        },
    }

    card = GeneratedCard.model_validate(payload)

    assert card.adjective_forms is not None
    assert card.adjective_forms.onverbuigbaar_example == "gouden ring"
