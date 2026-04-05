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
