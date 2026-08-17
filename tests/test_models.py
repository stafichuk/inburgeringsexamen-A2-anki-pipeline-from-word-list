import pytest
from pydantic import ValidationError

from app.models import (
    GeneratedCard,
    NounNumber,
    SourceConcept,
    VerbForms,
    matches_explicit_dutch_answer,
)


def test_explicit_dutch_answer_matching_allows_only_article_insertion() -> None:
    assert matches_explicit_dutch_answer("de kleding", "de kleding")
    assert matches_explicit_dutch_answer(" DE   KLEDING ", "de kleding")
    assert matches_explicit_dutch_answer("de kleding", "kleding")
    assert not matches_explicit_dutch_answer("de garderobe", "de kleding")
    assert not matches_explicit_dutch_answer("kleding", "de kleding")


def test_source_concept_expands_group_into_independent_answer_leaves() -> None:
    concept = SourceConcept(
        entry_id="clothes",
        dutch_answers=("de kleding", "de kleren"),
        translation_hint="одежда",
        topic="Kleding",
        lesson="Les 6",
        exam_level="A2",
    )

    source_items = concept.source_items()

    assert concept.source_text() == "de kleding | de kleren"
    assert concept.identity_key() == "explicit:clothes"
    assert [item.text for item in source_items] == ["de kleding", "de kleren"]
    assert [item.entry_id for item in source_items] == ["clothes", None]
    assert [item.answer_index for item in source_items] == [0, 1]
    assert all(item.concept_identity_key() == "explicit:clothes" for item in source_items)
    assert all(
        item.accepted_dutch_answers == ("de kleding", "de kleren")
        for item in source_items
    )
    assert "concept" not in source_items[0].model_dump()
    assert "answer_index" not in source_items[0].model_dump()


def test_source_concept_keeps_single_answer_as_a_standalone_source_item() -> None:
    concept = SourceConcept(
        entry_id="clothes",
        dutch_answers=("de kleding",),
        translation_hint="одежда",
    )

    source_item = concept.source_items()[0]

    assert source_item.entry_id == "clothes"
    assert source_item.concept is None
    assert source_item.answer_index == 0
    assert source_item.accepted_dutch_answers == ("de kleding",)
    assert source_item.concept_identity_key() == source_item.identity_key()


def test_source_concept_requires_hint_for_grouped_answers() -> None:
    with pytest.raises(ValidationError, match="grouped concepts must include a Russian translation hint"):
        SourceConcept(dutch_answers=("de kleding", "de kleren"))


def test_source_concept_rejects_normalized_duplicate_answers() -> None:
    with pytest.raises(ValidationError, match="Dutch answers must be unique"):
        SourceConcept(
            dutch_answers=("de kleding", " DE   KLEDING "),
            translation_hint="одежда",
        )


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
                "form": "school",
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
    assert card.noun_number == NounNumber.COUNTABLE


def test_generated_card_accepts_countable_noun_forms_without_article() -> None:
    payload = {
        "dutch_word": "de tante",
        "russian_translation": "тётя",
        "part_of_speech": "noun",
        "ipa_transcription": "ˈtɑn.tə",
        "lesson_topic": "De familie",
        "form_examples": [
            {
                "kind": "singular",
                "form": "tante",
                "example_sentence_nl": "Mijn tante woont in Amsterdam.",
                "example_sentence_ru": "Моя тётя живёт в Амстердаме.",
            },
            {
                "kind": "plural",
                "form": "tantes",
                "example_sentence_nl": "Mijn twee tantes komen op bezoek.",
                "example_sentence_ru": "Мои две тёти придут в гости.",
            },
        ],
        "tags": ["familie", "lesson-3"],
        "plural_form": "tantes",
        "front_hint": "тётя (множественное число?)",
        "verb_forms": None,
        "adjective_forms": None,
    }

    card = GeneratedCard.model_validate(payload)

    assert card.dutch_word == "de tante"
    assert card.plural_form == "tantes"


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
    assert card.noun_number == NounNumber.UNCOUNTABLE


def test_generated_card_accepts_plural_only_noun_without_singularizing_it() -> None:
    card = GeneratedCard.model_validate(
        {
            "dutch_word": "de kleren",
            "russian_translation": "одежда",
            "part_of_speech": "noun",
            "ipa_transcription": "/ˈkleː.rə(n)/",
            "lesson_topic": "Kleding",
            "form_examples": [
                {
                    "kind": "default",
                    "form": "kleren",
                    "example_sentence_nl": "Mijn kleren zijn schoon.",
                    "example_sentence_ru": "Моя одежда чистая.",
                }
            ],
            "tags": ["kleding"],
            "plural_form": None,
            "noun_number": "plural_only",
            "front_hint": "одежда",
            "verb_forms": None,
            "adjective_forms": None,
        }
    )

    assert card.dutch_word == "de kleren"
    assert card.noun_number == NounNumber.PLURAL_ONLY
    assert card.plural_form is None
    assert [example.kind.value for example in card.form_examples] == ["default"]


def test_generated_card_rejects_plural_form_for_plural_only_noun() -> None:
    with pytest.raises(ValidationError, match="plural_only nouns must not include plural_form"):
        GeneratedCard.model_validate(
            {
                "dutch_word": "de kleren",
                "russian_translation": "одежда",
                "part_of_speech": "noun",
                "ipa_transcription": "/ˈkleː.rə(n)/",
                "lesson_topic": "Kleding",
                "form_examples": [
                    {
                        "kind": "default",
                        "form": "kleren",
                        "example_sentence_nl": "Mijn kleren zijn schoon.",
                        "example_sentence_ru": "Моя одежда чистая.",
                    }
                ],
                "plural_form": "kleren",
                "noun_number": "plural_only",
                "front_hint": "одежда",
            }
        )


def test_generated_card_rejects_noun_number_for_non_noun() -> None:
    with pytest.raises(ValidationError, match="non-nouns must set noun_number to null"):
        GeneratedCard.model_validate(
            {
                "dutch_word": "gisteren",
                "russian_translation": "вчера",
                "part_of_speech": "adverb",
                "ipa_transcription": "/ˈɣɪs.tə.rə(n)/",
                "lesson_topic": "Tijd",
                "form_examples": [
                    {
                        "kind": "default",
                        "form": "gisteren",
                        "example_sentence_nl": "Gisteren werkte ik thuis.",
                        "example_sentence_ru": "Вчера я работал дома.",
                    }
                ],
                "noun_number": "uncountable",
            }
        )


def test_generated_card_accepts_uncountable_noun_singular_example_alias() -> None:
    payload = {
        "dutch_word": "de liefde",
        "russian_translation": "любовь",
        "part_of_speech": "noun",
        "ipa_transcription": "/də ˈlif.də/",
        "lesson_topic": "De familie",
        "form_examples": [
            {
                "kind": "singular",
                "form": "liefde",
                "example_sentence_nl": "De liefde in onze familie is heel sterk.",
                "example_sentence_ru": "Любовь в нашей семье очень сильная.",
            }
        ],
        "tags": ["familie", "A2", "inburgering", "spreken"],
        "plural_form": None,
        "front_hint": "любовь",
        "verb_forms": None,
        "adjective_forms": None,
    }

    card = GeneratedCard.model_validate(payload)

    assert [example.kind.value for example in card.form_examples] == ["default"]


def test_generated_card_enum_values_are_case_insensitive() -> None:
    payload = {
        "dutch_word": "DE MELK",
        "russian_translation": "молоко",
        "part_of_speech": "NOUN",
        "ipa_transcription": "mɛlk",
        "lesson_topic": "Eten en drinken",
        "form_examples": [
            {
                "kind": "DEFAULT",
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

    assert card.part_of_speech.value == "noun"
    assert card.form_examples[0].kind.value == "default"


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


def test_generated_card_rejects_countable_noun_plural_form_with_article() -> None:
    payload = {
        "dutch_word": "de tante",
        "russian_translation": "тётя",
        "part_of_speech": "noun",
        "ipa_transcription": "ˈtɑn.tə",
        "lesson_topic": "De familie",
        "form_examples": [
            {
                "kind": "singular",
                "form": "tante",
                "example_sentence_nl": "Mijn tante woont in Amsterdam.",
                "example_sentence_ru": "Моя тётя живёт в Амстердаме.",
            },
            {
                "kind": "plural",
                "form": "tantes",
                "example_sentence_nl": "Mijn twee tantes komen op bezoek.",
                "example_sentence_ru": "Мои две тёти придут в гости.",
            },
        ],
        "tags": ["familie", "lesson-3"],
        "plural_form": "de tantes",
        "front_hint": "тётя (множественное число?)",
        "verb_forms": None,
        "adjective_forms": None,
    }

    try:
        GeneratedCard.model_validate(payload)
    except ValidationError as exc:
        assert "plural_form must not include article" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("validation should have failed")


def test_generated_card_rejects_countable_noun_with_plural_headword() -> None:
    payload = {
        "dutch_word": "de ouders",
        "russian_translation": "родители",
        "part_of_speech": "noun",
        "ipa_transcription": "ˈʌu̯.dərs",
        "lesson_topic": "De familie",
        "form_examples": [
            {
                "kind": "singular",
                "form": "ouders",
                "example_sentence_nl": "Mijn ouders wonen dichtbij.",
                "example_sentence_ru": "Мои родители живут рядом.",
            },
            {
                "kind": "plural",
                "form": "ouders",
                "example_sentence_nl": "Veel ouders wachten buiten.",
                "example_sentence_ru": "Многие родители ждут снаружи.",
            },
        ],
        "tags": ["familie", "lesson-3"],
        "plural_form": "ouders",
        "front_hint": "родитель (множественное число?)",
        "verb_forms": None,
        "adjective_forms": None,
    }

    try:
        GeneratedCard.model_validate(payload)
    except ValidationError as exc:
        assert "singular" in str(exc)
        assert "plural_form" in str(exc)
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
                "form": "school",
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
                "form": "school",
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
                "form": "ik leer",
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
                "kind": "perfect_tense",
                "form": "heeft geleerd",
                "example_sentence_nl": "Hij heeft veel geleerd.",
                "example_sentence_ru": "Я многому научился.",
            },
        ],
        "tags": ["school", "verb"],
        "plural_form": None,
        "front_hint": None,
        "verb_forms": {
            "infinitive": "leren",
            "present_ik": "ik leer",
            "present_hij": "hij leert",
            "past_tense": "leerde",
            "perfect_tense": "heeft geleerd",
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
        "perfect_tense",
    ]


def test_generated_card_accepts_verb_with_preposition_without_literal_form_match() -> None:
    payload = {
        "dutch_word": "houden van",
        "russian_translation": "любить",
        "part_of_speech": "verb",
        "ipa_transcription": "/ˈɦɑu̯.də(n) vɑn/",
        "lesson_topic": "De familie",
        "form_examples": [
            {
                "kind": "present_tense",
                "form": "houden van",
                "example_sentence_nl": "Ik hou van mijn familie.",
                "example_sentence_ru": "Я люблю свою семью.",
            },
            {
                "kind": "past_tense",
                "form": "hielden van",
                "example_sentence_nl": "Wij hielden van onze opa.",
                "example_sentence_ru": "Мы любили нашего дедушку.",
            },
            {
                "kind": "perfect_tense",
                "form": "gehouden van",
                "example_sentence_nl": "Zij heeft altijd van haar broer gehouden.",
                "example_sentence_ru": "Она всегда любила своего брата.",
            },
        ],
        "tags": ["familie", "werkwoord"],
        "plural_form": None,
        "front_hint": None,
        "verb_forms": {
            "infinitive": "houden van",
            "present_ik": "ik hou van",
            "present_hij": "hij houdt van",
            "past_tense": "hield van",
            "perfect_tense": "heeft gehouden van",
            "separable_prefix": None,
            "conjugation_notes": "Onregelmatig: ik hou / hij houdt.",
        },
        "adjective_forms": None,
    }

    card = GeneratedCard.model_validate(payload)

    assert card.dutch_word == "houden van"


def test_generated_card_accepts_separable_verb_without_literal_form_match() -> None:
    payload = {
        "dutch_word": "oppassen",
        "russian_translation": "быть осторожным, присматривать",
        "part_of_speech": "verb",
        "ipa_transcription": "/ˈɔpɑsə(n)/",
        "lesson_topic": "De familie",
        "form_examples": [
            {
                "kind": "present_tense",
                "form": "past op",
                "example_sentence_nl": "Ik pas op mijn kleine zusje.",
                "example_sentence_ru": "Я присматриваю за своей младшей сестрой.",
            },
            {
                "kind": "past_tense",
                "form": "paste op",
                "example_sentence_nl": "Gisteren paste ik op de baby.",
                "example_sentence_ru": "Вчера я присматривал за ребёнком.",
            },
            {
                "kind": "perfect_tense",
                "form": "heeft opgepast",
                "example_sentence_nl": "Mijn broer heeft goed opgepast.",
                "example_sentence_ru": "Мой брат хорошо присмотрел.",
            },
        ],
        "tags": ["familie", "werkwoord", "scheidbaar"],
        "plural_form": None,
        "front_hint": None,
        "verb_forms": {
            "infinitive": "oppassen",
            "present_ik": "ik pas op",
            "present_hij": "hij past op",
            "past_tense": "paste op",
            "perfect_tense": "heeft opgepast",
            "separable_prefix": "op",
            "conjugation_notes": None,
        },
        "adjective_forms": None,
    }

    card = GeneratedCard.model_validate(payload)

    assert card.verb_forms is not None
    assert card.verb_forms.separable_prefix == "op"


def test_generated_card_rejects_verb_without_perfect_tense_example() -> None:
    payload = {
        "dutch_word": "leren",
        "russian_translation": "учиться",
        "part_of_speech": "verb",
        "ipa_transcription": "ˈleːrə(n)",
        "lesson_topic": "De school",
        "form_examples": [
            {
                "kind": "present_tense",
                "form": "ik leer",
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
            "present_ik": "ik leer",
            "present_hij": "hij leert",
            "past_tense": "leerde",
            "perfect_tense": "heeft geleerd",
        },
        "adjective_forms": None,
    }

    try:
        GeneratedCard.model_validate(payload)
    except ValidationError as exc:
        assert "verbs" in str(exc)
        assert "perfect_tense" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("validation should have failed")


def test_generated_card_rejects_verb_without_hij_present_tense_pronoun() -> None:
    payload = {
        "dutch_word": "leren",
        "russian_translation": "учиться",
        "part_of_speech": "verb",
        "ipa_transcription": "ˈleːrə(n)",
        "lesson_topic": "De school",
        "form_examples": [
            {
                "kind": "present_tense",
                "form": "ik leer",
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
                "kind": "perfect_tense",
                "form": "heeft geleerd",
                "example_sentence_nl": "Hij heeft veel geleerd.",
                "example_sentence_ru": "Я многому научился.",
            },
        ],
        "tags": ["school", "verb"],
        "plural_form": None,
        "front_hint": None,
        "verb_forms": {
            "infinitive": "leren",
            "present_ik": "ik leer",
            "present_hij": "leert",
            "past_tense": "leerde",
            "perfect_tense": "heeft geleerd",
        },
        "adjective_forms": None,
    }

    try:
        GeneratedCard.model_validate(payload)
    except ValidationError as exc:
        assert "present_hij" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("validation should have failed")


def test_verb_forms_reject_present_ik_without_expected_pronoun() -> None:
    try:
        VerbForms(
            infinitive="leren",
            present_ik="leer",
            present_hij="hij leert",
            past_tense="leerde",
            perfect_tense="heeft geleerd",
        )
    except ValidationError as exc:
        assert "present_ik" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("validation should have failed")


def test_verb_forms_reject_removed_legacy_fields() -> None:
    try:
        VerbForms.model_validate(
            {
                "infinitive": "leren",
                "present_tense": "ik leer; hij leert",
                "past_tense": "leerde",
                "past_participle": "geleerd",
                "perfect_example": "Ik heb Nederlands geleerd.",
            }
        )
    except ValidationError as exc:
        message = str(exc)
        assert "present_ik" in message
        assert "present_hij" in message
        assert "perfect_tense" in message
        assert "present_tense" in message
        assert "past_participle" in message
        assert "perfect_example" in message
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
        message = str(exc)
        assert "form 'mooi' must appear" in message
        assert "'Het huis is groot.'" in message
    else:  # pragma: no cover
        raise AssertionError("validation should have failed")
