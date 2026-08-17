from app.models import GeneratedCard, SourceConcept, SourceItem
from app.prompts import build_messages


def _accepted_adverb_card() -> GeneratedCard:
    return GeneratedCard.model_validate(
        {
            "dutch_word": "gisteren",
            "russian_translation": "УНИКАЛЬНЫЙ РУССКИЙ ПЕРЕВОД",
            "part_of_speech": "adverb",
            "ipa_transcription": "/UNIQUE-IPA/",
            "lesson_topic": "PRIVATE ACCEPTED TOPIC",
            "form_examples": [
                {
                    "kind": "default",
                    "form": "gisteren",
                    "example_sentence_nl": "Gisteren fietste Noor naar haar werk.",
                    "example_sentence_ru": "УНИКАЛЬНЫЙ ПЕРЕВОД ПРИМЕРА",
                }
            ],
            "tags": ["accepted-private-tag"],
            "plural_form": None,
            "front_hint": None,
            "verb_forms": None,
            "adjective_forms": None,
        }
    )


def _build_single_item_prompt(source_item: SourceItem) -> str:
    return build_messages([(7, source_item)], [])[1]["content"]


def test_prompt_coordinates_pending_items_with_dutch_only_accepted_context() -> None:
    pending_items = [
        (
            17,
            SourceItem(
                text="de neef",
                translation_hint="племянник",
                topic="Familie",
                lesson="Les 4",
                exam_level="A2",
            ),
        ),
        (29, SourceItem(text="fietsen", topic="Verkeer")),
    ]
    accepted_source = SourceItem(
        text="gisteren",
        translation_hint="PRIVATE SOURCE TRANSLATION",
        topic="PRIVATE SOURCE TOPIC",
        lesson="PRIVATE SOURCE LESSON",
        exam_level="PRIVATE SOURCE LEVEL",
    )

    messages = build_messages(
        pending_items,
        [(accepted_source, _accepted_adverb_card())],
    )
    user_prompt = messages[1]["content"]

    assert '"source_id": 17' in user_prompt
    assert '"source_id": 29' in user_prompt
    assert '"input_item": "de neef"' in user_prompt
    assert '"translation_hint": "племянник"' in user_prompt
    assert '"lesson": "Les 4"' in user_prompt

    assert '"input_item": "gisteren"' in user_prompt
    assert '"dutch_word": "gisteren"' in user_prompt
    assert '"form": "gisteren"' in user_prompt
    assert '"sentence_nl": "Gisteren fietste Noor naar haar werk."' in user_prompt
    assert "PRIVATE SOURCE TRANSLATION" not in user_prompt
    assert "PRIVATE SOURCE TOPIC" not in user_prompt
    assert "PRIVATE SOURCE LESSON" not in user_prompt
    assert "PRIVATE SOURCE LEVEL" not in user_prompt
    assert "УНИКАЛЬНЫЙ РУССКИЙ ПЕРЕВОД" not in user_prompt
    assert "УНИКАЛЬНЫЙ ПЕРЕВОД ПРИМЕРА" not in user_prompt
    assert "/UNIQUE-IPA/" not in user_prompt
    assert "PRIVATE ACCEPTED TOPIC" not in user_prompt
    assert "accepted-private-tag" not in user_prompt


def test_prompt_requests_deck_wide_diversity_and_softens_topic() -> None:
    user_prompt = _build_single_item_prompt(
        SourceItem(text="de jongen", topic="Het formulier en de agenda")
    )

    assert "soft theme, not as a mandatory setting" in user_prompt
    assert "never insert them merely to appear relevant" in user_prompt
    assert "accepted examples and all examples in this response as one corpus" in user_prompt
    assert "Choose contexts yourself and distribute them naturally" in user_prompt
    assert "Do not reuse the same target-masked sentence frame" in user_prompt
    assert "Correct meaning, natural Dutch, and A2 clarity take priority over novelty" in user_prompt


def test_prompt_requires_output_to_match_pending_source_ids() -> None:
    user_prompt = build_messages(
        [
            (3, SourceItem(text="leren")),
            (41, SourceItem(text="de school")),
        ],
        [],
    )[1]["content"]

    assert "one object for every source_id in the unresolved input list" in user_prompt
    assert "Do not return accepted reference cards or invent source IDs" in user_prompt
    assert "echo source_id, input_item, and translation_hint exactly" in user_prompt
    assert "Do not normalize, correct, translate" in user_prompt
    assert "echo it as JSON null when it is null" in user_prompt
    assert (
        '{"source_id": 1, "input_item": "exact original input_item", '
        '"translation_hint": null, "card": {...}}'
        in user_prompt
    )


def test_prompt_passes_translation_hint_as_strict_sense_constraint() -> None:
    source_item = SourceItem(text="de neef", translation_hint="племянник", topic="Familie")

    user_prompt = _build_single_item_prompt(source_item)

    assert '"input_item": "de neef"' in user_prompt
    assert '"translation_hint": "племянник"' in user_prompt
    assert "strict sense constraint" in user_prompt
    assert "alternative meanings" in user_prompt
    assert "de neef - племянник" not in user_prompt


def test_prompt_keeps_grouped_answers_as_separate_explicit_rows() -> None:
    source_items = SourceConcept(
        entry_id="clothes",
        dutch_answers=("de kleding", "de kleren"),
        translation_hint="одежда",
        topic="Kleding",
    ).source_items()

    user_prompt = build_messages(
        [(11, source_items[0]), (12, source_items[1])],
        [],
    )[1]["content"]

    assert user_prompt.count('"accepted_dutch_answers"') == 2
    assert '"input_item": "de kleding"' in user_prompt
    assert '"input_item": "de kleren"' in user_prompt
    assert '"de kleding"' in user_prompt
    assert '"de kleren"' in user_prompt
    assert "Never invent or replace an accepted Dutch answer" in user_prompt
    assert "merge sibling answers into dutch_word" in user_prompt
    assert "add a synonym that is not listed" in user_prompt


def test_prompt_normalizes_plural_noun_inputs_to_singular_lemma() -> None:
    source_item = SourceItem(text="de ouders", topic="Familie")

    user_prompt = _build_single_item_prompt(source_item)

    assert '"input_item": "de ouders"' in user_prompt
    assert "already a plural noun" in user_prompt
    assert 'dutch_word "de ouder"' in user_prompt
    assert 'noun_number "countable"' in user_prompt
    assert 'plural_form "ouders"' in user_prompt


def test_prompt_preserves_lexicalized_plural_only_noun() -> None:
    user_prompt = _build_single_item_prompt(SourceItem(text="de kleren", topic="Kleding"))

    assert 'input "de kleren" must keep dutch_word "de kleren"' in user_prompt
    assert 'noun_number to "plural_only"' in user_prompt
    assert "set plural_form to null" in user_prompt
    assert "one default form_example using the plural-only headword" in user_prompt


def test_prompt_distinguishes_article_bearing_noun_from_bare_example_forms() -> None:
    user_prompt = _build_single_item_prompt(SourceItem(text="het hoofd", topic="Het lichaam"))

    assert 'dutch_word "het hoofd"' in user_prompt
    assert 'singular form "hoofd" in "Mijn hoofd doet pijn."' in user_prompt
    assert "Never include de or het in a noun form_examples form" in user_prompt


def test_prompt_keeps_countable_noun_front_hint_free_of_plural_answer() -> None:
    user_prompt = _build_single_item_prompt(SourceItem(text="het hoofd", topic="Het lichaam"))

    assert "front_hint must contain only the Russian meaning or sense hint" in user_prompt
    assert "Do not include Dutch text, plural_form, or plural-recall wording in front_hint" in user_prompt
    assert "The application adds the plural-recall question" in user_prompt


def test_prompt_teaches_month_names_as_de_nouns() -> None:
    source_item = SourceItem(text="januari", topic="Het formulier en de agenda")

    user_prompt = _build_single_item_prompt(source_item)

    assert "Month names are Dutch de-nouns" in user_prompt
    assert 'dutch_word to "de januari"' in user_prompt
    assert 'or "de december"' in user_prompt
    assert "Use the bare month name in form_examples" in user_prompt


def test_prompt_includes_validation_feedback_for_repair_attempt() -> None:
    validation_error = (
        "source_id 8: nouns must include article 'de' or 'het' in dutch_word\n"
        "source_id 13: missing from response"
    )

    messages = build_messages(
        [
            (8, SourceItem(text="januari", topic="Het formulier en de agenda")),
            (13, SourceItem(text="de vriend", topic="Familie")),
        ],
        [(SourceItem(text="gisteren"), _accepted_adverb_card())],
        validation_error=validation_error,
    )

    assert len(messages) == 3
    repair_prompt = messages[2]["content"]
    assert "previous batch attempt left some requested source IDs unresolved" in repair_prompt
    assert "already contains only the cards that still need to be generated" in repair_prompt
    assert "Echo the exact input_item and translation_hint paired with each source_id" in repair_prompt
    assert "Include translation_hint explicitly as JSON null" in repair_prompt
    assert validation_error in repair_prompt
