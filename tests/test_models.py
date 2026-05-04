from pydantic import ValidationError

from app.models import GeneratedCard


def test_generated_card_accepts_valid_countable_noun_payload() -> None:
    payload = {
        "dutch_word": "de school",
        "russian_translation": "школа",
        "part_of_speech": "noun",
        "ipa_transcription": "sxoːl",
        "lesson_topic": "De school",
        "form_examples": [
            {
                "kind": "singular",
                "form": "de school",
                "example_sentence_nl": "De school is dichtbij.",
                "example_sentence_ru": "Школа находится рядом.",
            },
            {
                "kind": "plural",
                "form": "scholen",
                "example_sentence_nl": "De scholen zijn dichtbij.",
                "example_sentence_ru": "Школы находятся рядом.",
            },
        ],
        "tags": ["school", "lesson-3"],
        "plural_form": "scholen",
        "front_hint": "школа (множественное число?)",
        "verb_forms": None,
        "adjective_forms": None,
    }

    card = GeneratedCard.model_validate(payload)
    assert card.plural_form == "scholen"


def test_generated_card_accepts_uncountable_noun_without_plural_prompt() -> None:
    payload = {
        "dutch_word": "de melk",
        "russian_translation": "молоко",
        "part_of_speech": "noun",
        "ipa_transcription": "mɛlk",
        "lesson_topic": "Eten en drinken",
        "form_examples": [
            {
                "kind": "default",
                "form": "melk",
                "example_sentence_nl": "Ik drink melk.",
                "example_sentence_ru": "Я пью молоко.",
            }
        ],
        "tags": ["food"],
        "plural_form": None,
        "front_hint": "молоко",
        "verb_forms": None,
        "adjective_forms": None,
    }

    card = GeneratedCard.model_validate(payload)

    assert card.plural_form is None
    assert card.front_hint == "молоко"


def test_generated_card_rejects_uncountable_noun_with_plural_prompt() -> None:
    payload = {
        "dutch_word": "de melk",
        "russian_translation": "молоко",
        "part_of_speech": "noun",
        "ipa_transcription": "mɛlk",
        "lesson_topic": "Eten en drinken",
        "form_examples": [
            {
                "kind": "default",
                "form": "melk",
                "example_sentence_nl": "Ik drink melk.",
                "example_sentence_ru": "Я пью молоко.",
            }
        ],
        "tags": ["food"],
        "plural_form": None,
        "front_hint": "молоко (множественное число?)",
        "verb_forms": None,
        "adjective_forms": None,
    }

    try:
        GeneratedCard.model_validate(payload)
    except ValidationError as exc:
        assert "plural recall" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("validation should have failed")


def test_generated_card_rejects_countable_noun_without_plural_example() -> None:
    payload = {
        "dutch_word": "de school",
        "russian_translation": "школа",
        "part_of_speech": "noun",
        "ipa_transcription": "sxoːl",
        "lesson_topic": "De school",
        "form_examples": [
            {
                "kind": "singular",
                "form": "de school",
                "example_sentence_nl": "De school is dichtbij.",
                "example_sentence_ru": "Школа находится рядом.",
            }
        ],
        "tags": ["school", "lesson-3"],
        "plural_form": "scholen",
        "front_hint": "школа (множественное число?)",
        "verb_forms": None,
        "adjective_forms": None,
    }

    try:
        GeneratedCard.model_validate(payload)
    except ValidationError as exc:
        assert "countable nouns" in str(exc)
        assert "plural" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("validation should have failed")


def test_generated_card_rejects_noun_without_article_in_word() -> None:
    payload = {
        "dutch_word": "school",
        "russian_translation": "школа",
        "part_of_speech": "noun",
        "ipa_transcription": "sxoːl",
        "lesson_topic": "De school",
        "form_examples": [
            {
                "kind": "singular",
                "form": "de school",
                "example_sentence_nl": "De school is dichtbij.",
                "example_sentence_ru": "Школа находится рядом.",
            },
            {
                "kind": "plural",
                "form": "scholen",
                "example_sentence_nl": "De scholen zijn dichtbij.",
                "example_sentence_ru": "Школы находятся рядом.",
            },
        ],
        "tags": ["school", "lesson-3"],
        "plural_form": "scholen",
        "front_hint": "школа (множественное число?)",
        "verb_forms": None,
        "adjective_forms": None,
    }

    try:
        GeneratedCard.model_validate(payload)
    except ValidationError as exc:
        assert "in dutch_word" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("validation should have failed")


def test_generated_card_accepts_verb_with_three_form_examples() -> None:
    payload = {
        "dutch_word": "leren",
        "russian_translation": "учиться",
        "part_of_speech": "verb",
        "ipa_transcription": "ˈleːrə(n)",
        "lesson_topic": "De school",
        "form_examples": [
            {
                "kind": "present_tense",
                "form": "leer",
                "example_sentence_nl": "Ik leer Nederlands op school.",
                "example_sentence_ru": "Я учу нидерландский в школе.",
            },
            {
                "kind": "past_tense",
                "form": "leerde",
                "example_sentence_nl": "Ik leerde gisteren nieuwe woorden.",
                "example_sentence_ru": "Вчера я учил новые слова.",
            },
            {
                "kind": "past_participle",
                "form": "geleerd",
                "example_sentence_nl": "Ik heb veel geleerd.",
                "example_sentence_ru": "Я многому научился.",
            },
        ],
        "tags": ["school", "verb"],
        "plural_form": None,
        "front_hint": None,
        "verb_forms": {
            "infinitive": "leren",
            "present_tense": "ik leer, jij leert, hij leert",
            "past_tense": "leerde, leerden",
            "past_participle": "geleerd",
            "perfect_example": "Ik heb Nederlands geleerd.",
            "separable_prefix": None,
            "conjugation_notes": "regular weak verb",
        },
        "adjective_forms": None,
    }

    card = GeneratedCard.model_validate(payload)

    assert card.verb_forms is not None
    assert [example.kind.value for example in card.ordered_form_examples()] == [
        "present_tense",
        "past_tense",
        "past_participle",
    ]


def test_generated_card_rejects_verb_without_past_participle_example() -> None:
    payload = {
        "dutch_word": "leren",
        "russian_translation": "учиться",
        "part_of_speech": "verb",
        "ipa_transcription": "ˈleːrə(n)",
        "lesson_topic": "De school",
        "form_examples": [
            {
                "kind": "present_tense",
                "form": "leer",
                "example_sentence_nl": "Ik leer Nederlands op school.",
                "example_sentence_ru": "Я учу нидерландский в школе.",
            },
            {
                "kind": "past_tense",
                "form": "leerde",
                "example_sentence_nl": "Ik leerde gisteren nieuwe woorden.",
                "example_sentence_ru": "Вчера я учил новые слова.",
            },
        ],
        "tags": ["school", "verb"],
        "plural_form": None,
        "front_hint": None,
        "verb_forms": {
            "infinitive": "leren",
            "present_tense": "ik leer, jij leert, hij leert",
            "past_tense": "leerde, leerden",
            "past_participle": "geleerd",
        },
        "adjective_forms": None,
    }

    try:
        GeneratedCard.model_validate(payload)
    except ValidationError as exc:
        assert "verbs" in str(exc)
        assert "past_participle" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("validation should have failed")


def test_generated_card_accepts_regular_adjective_with_base_and_e_form_examples() -> None:
    payload = {
        "dutch_word": "mooi",
        "russian_translation": "красивый",
        "part_of_speech": "adjective",
        "ipa_transcription": "moːi",
        "lesson_topic": "Het huis",
        "form_examples": [
            {
                "kind": "base_form",
                "form": "mooi",
                "example_sentence_nl": "Ik zie een mooi huis.",
                "example_sentence_ru": "Я вижу красивый дом.",
            },
            {
                "kind": "e_form",
                "form": "mooie",
                "example_sentence_nl": "Het mooie huis is groot.",
                "example_sentence_ru": "Красивый дом большой.",
            },
        ],
        "tags": ["house", "adjective"],
        "plural_form": None,
        "front_hint": None,
        "verb_forms": None,
        "adjective_forms": None,
    }

    card = GeneratedCard.model_validate(payload)

    assert card.adjective_forms is None
    assert [example.form for example in card.ordered_form_examples()] == ["mooi", "mooie"]


def test_generated_card_accepts_single_form_adjective_example() -> None:
    payload = {
        "dutch_word": "gouden",
        "russian_translation": "золотой",
        "part_of_speech": "adjective",
        "ipa_transcription": "ˈɣʌudə(n)",
        "lesson_topic": "Kleding",
        "form_examples": [
            {
                "kind": "single_form",
                "form": "gouden",
                "example_sentence_nl": "De gouden ring is mooi.",
                "example_sentence_ru": "Золотое кольцо красивое.",
            }
        ],
        "tags": ["clothing", "adjective"],
        "plural_form": None,
        "front_hint": None,
        "verb_forms": None,
        "adjective_forms": {
            "onverbuigbaar_example": "de gouden ring",
            "learner_note": "Stofadjectief op -en.",
        },
    }

    card = GeneratedCard.model_validate(payload)

    assert card.adjective_forms is not None
    assert card.form_examples[0].form == "gouden"


def test_generated_card_rejects_regular_adjective_with_adjective_forms() -> None:
    payload = {
        "dutch_word": "mooi",
        "russian_translation": "красивый",
        "part_of_speech": "adjective",
        "ipa_transcription": "moːi",
        "lesson_topic": "Het huis",
        "form_examples": [
            {
                "kind": "base_form",
                "form": "mooi",
                "example_sentence_nl": "Ik zie een mooi huis.",
                "example_sentence_ru": "Я вижу красивый дом.",
            },
            {
                "kind": "e_form",
                "form": "mooie",
                "example_sentence_nl": "Het mooie huis is groot.",
                "example_sentence_ru": "Красивый дом большой.",
            },
        ],
        "tags": ["house", "adjective"],
        "plural_form": None,
        "front_hint": None,
        "verb_forms": None,
        "adjective_forms": {
            "onverbuigbaar_example": "mooie huis",
        },
    }

    try:
        GeneratedCard.model_validate(payload)
    except ValidationError as exc:
        assert "regular adjectives" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("validation should have failed")


def test_form_example_rejects_sentence_without_the_form() -> None:
    payload = {
        "dutch_word": "mooi",
        "russian_translation": "красивый",
        "part_of_speech": "adjective",
        "ipa_transcription": "moːi",
        "lesson_topic": "Het huis",
        "form_examples": [
            {
                "kind": "base_form",
                "form": "mooi",
                "example_sentence_nl": "Het huis is groot.",
                "example_sentence_ru": "Дом большой.",
            },
            {
                "kind": "e_form",
                "form": "mooie",
                "example_sentence_nl": "Het mooie huis is groot.",
                "example_sentence_ru": "Красивый дом большой.",
            },
        ],
        "tags": ["house", "adjective"],
        "plural_form": None,
        "front_hint": None,
        "verb_forms": None,
        "adjective_forms": None,
    }

    try:
        GeneratedCard.model_validate(payload)
    except ValidationError as exc:
        assert "form must appear" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("validation should have failed")
