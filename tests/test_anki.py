import hashlib
import json
from pathlib import Path
import sqlite3
import zipfile

import genanki
import pytest

from app.anki import (
    NOTE_FIELDS,
    PLURAL_PROMPT_SUFFIX,
    NoteAudio,
    VerbFormAudio,
    build_deck_package,
    build_example_slot_fields,
    build_front,
    build_grouped_note,
    build_note,
    build_note_guid,
    build_variant_html,
    build_verb_form_fields,
    create_note_model,
    format_adjective_forms,
    format_verb_forms,
)
from app.config import DeckSettings
from app.models import AdjectiveForms, GeneratedCard, SourceConcept, SourceItem, VerbForms


def make_card(word: str = "leren") -> GeneratedCard:
    present_ik = "ik leer" if word == "leren" else f"ik {word}"
    present_hij = "hij leert" if word == "leren" else f"hij {word}t"
    return GeneratedCard(
        dutch_word=word,
        russian_translation="учиться",
        part_of_speech="verb",
        ipa_transcription="ˈleːrə(n)",
        lesson_topic="De school",
        form_examples=[
            {
                "kind": "present_tense",
                "form": present_ik,
                "example_sentence_nl": f"{present_ik.capitalize()} Nederlands op school.",
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
                "example_sentence_nl": "Hij heeft Nederlands geleerd.",
                "example_sentence_ru": "Я выучил нидерландский.",
            },
        ],
        tags=["school", "verb"],
        verb_forms=VerbForms(
            infinitive=word,
            present_ik=present_ik,
            present_hij=present_hij,
            past_tense="leerde",
            perfect_tense="heeft geleerd",
            conjugation_notes="regular weak verb",
        ),
    )


def make_countable_noun_card() -> GeneratedCard:
    return GeneratedCard(
        dutch_word="de school",
        russian_translation="школа",
        part_of_speech="noun",
        ipa_transcription="sxoːl",
        lesson_topic="De school",
        form_examples=[
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
        tags=["school"],
        plural_form="scholen",
        front_hint="школа (множественное число?)",
    )


def make_clothing_concept_and_cards() -> tuple[
    SourceConcept,
    list[SourceItem],
    list[GeneratedCard],
]:
    concept = SourceConcept(
        entry_id="clothes",
        dutch_answers=("de kleding", "de kleren"),
        translation_hint="одежда",
        topic="Kleding",
        lesson="Les 1",
        exam_level="A2",
    )
    cards = [
        GeneratedCard(
            dutch_word="de kleding",
            russian_translation="одежда",
            part_of_speech="noun",
            ipa_transcription="ˈkleː.dɪŋ",
            lesson_topic="Kleding",
            form_examples=[
                {
                    "kind": "default",
                    "form": "kleding",
                    "example_sentence_nl": "Ik koop nieuwe kleding.",
                    "example_sentence_ru": "Я покупаю новую одежду.",
                }
            ],
            noun_number="uncountable",
            front_hint="одежда",
        ),
        GeneratedCard(
            dutch_word="de kleren",
            russian_translation="одежда",
            part_of_speech="noun",
            ipa_transcription="ˈkleː.rə(n)",
            lesson_topic="Kleding",
            form_examples=[
                {
                    "kind": "default",
                    "form": "kleren",
                    "example_sentence_nl": "Ik koop nieuwe kleren.",
                    "example_sentence_ru": "Я покупаю новую одежду.",
                }
            ],
            noun_number="plural_only",
            front_hint="одежда",
        ),
    ]
    return concept, concept.source_items(), cards


def test_deck_generation_writes_apkg(tmp_path: Path) -> None:
    settings = DeckSettings()
    source_item = SourceItem(text="leren", topic="De school", lesson="Lesson 3", exam_level="A2")
    card = make_card()
    output_path = tmp_path / "school.apkg"

    build_deck_package([(source_item, card)], output_path, "Lesson 3 - De school", settings)

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_note_guid_includes_translation_hint() -> None:
    nephew = SourceItem(text="de neef", translation_hint="племянник", topic="Familie", lesson="Les 1")
    cousin = SourceItem(text="de neef", translation_hint="двоюродный брат", topic="Familie", lesson="Les 1")
    plain = SourceItem(text="de neef", topic="Familie", lesson="Les 1")

    assert build_note_guid(nephew) != build_note_guid(cousin)
    assert build_note_guid(nephew) != build_note_guid(plain)
    assert build_note_guid(cousin) != build_note_guid(plain)


def test_implicit_note_guid_normalizes_word_and_hint_like_identity() -> None:
    canonical = SourceItem(
        text="de neef",
        translation_hint="двоюродный брат",
        topic="Familie",
        lesson="Les 1",
    )
    equivalent = SourceItem(
        text="DE   NEEF",
        translation_hint="ДВОЮРОДНЫЙ\tБРАТ",
        topic="Familie",
        lesson="Les 1",
    )

    assert canonical.identity_key() == equivalent.identity_key()
    assert build_note_guid(canonical) == build_note_guid(equivalent)


def test_implicit_note_guid_preserves_old_seed_for_normalized_item() -> None:
    source_item = SourceItem(
        text="de neef",
        translation_hint="племянник",
        topic="Familie",
        lesson="Les 1",
    )
    old_seed = "de neef|племянник|Familie|Les 1"

    assert build_note_guid(source_item) == hashlib.md5(old_seed.encode("utf-8")).hexdigest()


def test_explicit_id_preserves_note_guid_across_word_correction() -> None:
    original = SourceItem(entry_id="friend", text="de vrient", topic="Familie", lesson="Les 1")
    corrected = SourceItem(entry_id="friend", text="de vriend", topic="Familie", lesson="Les 1")

    assert build_note_guid(original) == build_note_guid(corrected)


def test_grouped_note_guid_uses_shared_explicit_concept_id() -> None:
    original = SourceConcept(
        entry_id="clothes",
        dutch_answers=("de kleding", "de kleren"),
        translation_hint="одежда",
        topic="Kleding",
        lesson="Les 1",
    )
    revised = SourceConcept(
        entry_id="clothes",
        dutch_answers=("de kleren", "de kleding", "de garderobe"),
        translation_hint="одежда",
        topic="Kleding",
        lesson="Les 1",
    )

    original_items = original.source_items()
    revised_items = revised.source_items()

    assert len({build_note_guid(item) for item in original_items}) == 1
    assert len({build_note_guid(item) for item in revised_items}) == 1
    assert build_note_guid(original_items[0]) == build_note_guid(revised_items[0])


def test_implicit_grouped_note_guid_preserves_first_standalone_answer() -> None:
    standalone = SourceItem(
        text="de kleding",
        translation_hint="одежда",
        topic="Kleding",
        lesson="Les 1",
    )
    concept = SourceConcept(
        dutch_answers=("de kleding", "de kleren"),
        translation_hint="одежда",
        topic="Kleding",
        lesson="Les 1",
    )
    grouped_items = concept.source_items()

    assert len({build_note_guid(item) for item in grouped_items}) == 1
    assert build_note_guid(grouped_items[0]) == build_note_guid(standalone)


def test_note_model_contains_expected_fields() -> None:
    model = create_note_model(DeckSettings())
    model_field_names = [field["name"] for field in model.fields]
    assert model_field_names == NOTE_FIELDS
    assert "Example_NL" not in model_field_names
    assert "Example_RU" not in model_field_names
    assert "Example_Audio" not in model_field_names
    assert "Verb_Forms" not in model_field_names
    assert "Plural_Audio" in model_field_names
    assert "Verb_Infinitive" in model_field_names
    assert "Verb_Infinitive_Audio" in model_field_names
    assert "Verb_Present_Ik" in model_field_names
    assert "Verb_Present_Ik_Audio" in model_field_names
    assert "Verb_Present_Hij" in model_field_names
    assert "Verb_Present_Hij_Audio" in model_field_names
    assert "Verb_Past" in model_field_names
    assert "Verb_Past_Audio" in model_field_names
    assert "Verb_Perfect" in model_field_names
    assert "Verb_Perfect_Audio" in model_field_names
    assert "Example_1_Form" in model_field_names
    assert "Example_3_Audio" in model_field_names


def test_note_model_template_matches_updated_layout() -> None:
    model = create_note_model(DeckSettings())
    template = model.templates[0]["afmt"]
    css = model.css

    assert "Woordsoort:" not in template
    assert "{{#IPA}}" in template
    assert "{{^IPA}}" in template
    assert "{{POS}}" in template
    assert "{{Article}}" not in template
    assert "Voorbeelden:" in template
    assert "{{Example_NL}}" not in template
    assert "{{Example_RU}}" not in template
    assert "{{Example_Audio}}" not in template
    assert "{{Example_1_Form}}" in template
    assert "{{Example_1_NL}}{{Example_1_Audio}}" in template
    assert "{{Example_2_NL}}{{Example_2_Audio}}" in template
    assert "{{Example_3_NL}}{{Example_3_Audio}}" in template
    assert "Werkwoordsvormen:" in template
    assert "{{Verb_Forms}}" not in template
    assert "Infinitive:" not in template
    assert "{{Verb_Infinitive}}{{Verb_Infinitive_Audio}}" not in template
    assert "{{Verb_Present_Ik}}{{Verb_Present_Ik_Audio}}" in template
    assert "{{Verb_Present_Hij}}{{Verb_Present_Hij_Audio}}" in template
    assert "{{Verb_Past}}{{Verb_Past_Audio}}" in template
    assert "{{Verb_Perfect}}{{Verb_Perfect_Audio}}" in template
    assert "Bijvoeglijk naamwoord:" in template
    assert "Lesson:" not in template
    assert "Topic:" not in template
    assert "Article:" not in template
    assert "(meervoud {{Plural}}{{Plural_Audio}})" in template
    assert "color: #6b1d1d;" not in css


def test_build_front_does_not_add_plural_prompt_for_uncountable_noun() -> None:
    card = GeneratedCard(
        dutch_word="de melk",
        russian_translation="молоко",
        part_of_speech="noun",
        ipa_transcription="mɛlk",
        lesson_topic="Eten en drinken",
        form_examples=[
            {
                "kind": "default",
                "form": "melk",
                "example_sentence_nl": "Ik drink melk.",
                "example_sentence_ru": "Я пью молоко.",
            }
        ],
        tags=["food"],
        plural_form=None,
        front_hint="молоко",
    )

    assert build_front(card) == "молоко"


def test_build_front_hides_plural_leaked_by_countable_noun_hint() -> None:
    card = make_countable_noun_card().model_copy(
        update={
            "dutch_word": "het hoofd",
            "russian_translation": "голова",
            "plural_form": "hoofden",
            "front_hint": "голова (мн. ч. — hoofden)",
        }
    )

    assert card.front_hint == "голова (мн. ч. — hoofden)"
    front = build_front(card)

    assert front == f"голова{PLURAL_PROMPT_SUFFIX}"
    assert "hoofden" not in front


def test_build_front_preserves_russian_singular_hint_for_plural_translation() -> None:
    card = make_countable_noun_card().model_copy(
        update={
            "dutch_word": "de ouder",
            "russian_translation": "родители",
            "plural_form": "ouders",
            "front_hint": "родитель (множественное число?)",
        }
    )

    assert build_front(card) == f"родитель{PLURAL_PROMPT_SUFFIX}"


def test_build_front_falls_back_when_hint_contains_only_plural_answer() -> None:
    card = make_countable_noun_card().model_copy(
        update={
            "dutch_word": "de voet",
            "russian_translation": "стопа",
            "plural_form": "voeten",
            "front_hint": "мн.ч.: voeten",
        }
    )

    assert build_front(card) == f"стопа{PLURAL_PROMPT_SUFFIX}"


def test_build_grouped_note_uses_one_russian_front_and_complete_equal_blocks(
    tmp_path: Path,
) -> None:
    concept, source_items, cards = make_clothing_concept_and_cards()
    model = create_note_model(DeckSettings())
    reversed_generated_order = list(reversed(list(zip(source_items, cards, strict=True))))
    note = build_grouped_note(
        model,
        reversed_generated_order,
        audio=(
            NoteAudio(
                word_audio=tmp_path / "kleding.mp3",
                example_audios=(tmp_path / "kleding-example.mp3",),
            ),
            NoteAudio(
                word_audio=tmp_path / "kleren.mp3",
                example_audios=(tmp_path / "kleren-example.mp3",),
            ),
        ),
    )

    assert note.guid == build_note_guid(source_items[0])
    assert note.fields[NOTE_FIELDS.index("Front")] == "одежда"
    assert PLURAL_PROMPT_SUFFIX not in note.fields[NOTE_FIELDS.index("Front")]
    assert note.fields[NOTE_FIELDS.index("Word_NL")] == "de kleding<br>de kleren"
    assert note.fields[NOTE_FIELDS.index("Translation_RU")] == "одежда"
    assert note.fields[NOTE_FIELDS.index("IPA")] == ""
    assert note.fields[NOTE_FIELDS.index("SourceWord")] == concept.source_text()

    variants_html = note.fields[NOTE_FIELDS.index("POS")]
    assert variants_html.count('<section class="variant">') == 2
    assert variants_html.index("de kleding") < variants_html.index("de kleren")
    assert "ˈkleː.dɪŋ" in variants_html
    assert "ˈkleː.rə(n)" in variants_html
    assert "[sound:kleding.mp3]" in variants_html
    assert "[sound:kleren.mp3]" in variants_html
    assert "Ik koop nieuwe kleding." in variants_html
    assert "[sound:kleding-example.mp3]" in variants_html
    assert "Ik koop nieuwe kleren." in variants_html
    assert "[sound:kleren-example.mp3]" in variants_html

    for field_name in (
        "Plural",
        "Plural_Audio",
        "Verb_Infinitive",
        "Adjective_Forms",
        "Word_Audio",
        "Example_1_Form",
        "Example_1_NL",
        "Example_1_RU",
        "Example_1_Audio",
    ):
        assert note.fields[NOTE_FIELDS.index(field_name)] == ""


def test_grouped_variant_html_escapes_generated_text() -> None:
    _, _, cards = make_clothing_concept_and_cards()
    unsafe_example = cards[0].form_examples[0].model_copy(
        update={"example_sentence_nl": "Ik koop <b>kleding</b>."}
    )
    unsafe_card = cards[0].model_copy(
        update={
            "dutch_word": "de <script>kleding</script>",
            "ipa_transcription": '<img src=x onerror="alert(1)">',
            "form_examples": [unsafe_example],
        }
    )

    rendered = build_variant_html(unsafe_card)

    assert '<section class="variant">' in rendered
    assert "<script>" not in rendered
    assert "<img " not in rendered
    assert "<b>kleding</b>" not in rendered
    assert "&lt;script&gt;kleding&lt;/script&gt;" in rendered
    assert "&lt;img src=x onerror=&quot;alert(1)&quot;&gt;" in rendered
    assert "Ik koop &lt;b&gt;kleding&lt;/b&gt;." in rendered


def test_grouped_note_rejects_truncated_variant_audio() -> None:
    _, source_items, cards = make_clothing_concept_and_cards()

    with pytest.raises(ValueError, match="expected 2, got 1"):
        build_grouped_note(
            create_note_model(DeckSettings()),
            list(zip(source_items, cards, strict=True)),
            audio=(NoteAudio(),),
        )


def test_grouped_note_rejects_card_that_replaces_an_authored_answer() -> None:
    _, source_items, cards = make_clothing_concept_and_cards()
    cards[0] = cards[0].model_copy(update={"dutch_word": "de garderobe"})

    with pytest.raises(
        ValueError,
        match="expected 'de kleding', got 'de garderobe'",
    ):
        build_grouped_note(
            create_note_model(DeckSettings()),
            list(zip(source_items, cards, strict=True)),
        )


def test_grouped_note_suppresses_plural_question_for_countable_variant() -> None:
    _, _, clothing_cards = make_clothing_concept_and_cards()
    concept = SourceConcept(
        dutch_answers=("de school", "de kleren"),
        translation_hint="школа или одежда",
    )

    note = build_grouped_note(
        create_note_model(DeckSettings()),
        list(
            zip(
                concept.source_items(),
                [make_countable_noun_card(), clothing_cards[1]],
                strict=True,
            )
        ),
    )

    assert note.fields[NOTE_FIELDS.index("Front")] == "школа или одежда"
    assert PLURAL_PROMPT_SUFFIX not in note.fields[NOTE_FIELDS.index("Front")]
    assert "(meervoud scholen)" in note.fields[NOTE_FIELDS.index("POS")]


def test_grouped_note_renders_more_answers_than_legacy_example_slot_count() -> None:
    answers = ("leren", "studeren", "oefenen", "trainen")
    concept = SourceConcept(
        entry_id="study",
        dutch_answers=answers,
        translation_hint="учиться",
        topic="School",
        lesson="Les 1",
    )
    cards = [make_card(answer) for answer in answers]

    note = build_grouped_note(
        create_note_model(DeckSettings()),
        list(zip(concept.source_items(), cards, strict=True)),
    )
    variants_html = note.fields[NOTE_FIELDS.index("POS")]

    assert variants_html.count('<section class="variant">') == 4
    positions = [variants_html.index(answer) for answer in answers]
    assert positions == sorted(positions)


def test_build_note_includes_article_in_word_field_for_nouns() -> None:
    source_item = SourceItem(text="school", topic="De school", lesson="Lesson 3", exam_level="A2")
    model = create_note_model(DeckSettings())
    card = GeneratedCard(
        dutch_word="de school",
        russian_translation="школа",
        part_of_speech="noun",
        ipa_transcription="sxoːl",
        lesson_topic="De school",
        form_examples=[
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
        tags=["school"],
        plural_form="scholen",
        front_hint="школа (множественное число?)",
    )

    note = build_note(model, source_item, card)

    assert "Article" not in NOTE_FIELDS
    assert note.fields[NOTE_FIELDS.index("Word_NL")] == "de school"


def test_build_note_keeps_single_answer_in_legacy_editable_fields() -> None:
    source_item = SourceItem(
        text="leren",
        translation_hint="учиться",
        topic="De school",
        lesson="Lesson 3",
    )
    card = make_card()

    note = build_note(create_note_model(DeckSettings()), source_item, card)

    assert len(note.fields) == len(NOTE_FIELDS)
    assert note.fields[NOTE_FIELDS.index("Word_NL")] == "leren"
    assert note.fields[NOTE_FIELDS.index("IPA")] == "ˈleːrə(n)"
    assert note.fields[NOTE_FIELDS.index("POS")] == "werkwoord"
    assert note.fields[NOTE_FIELDS.index("Verb_Infinitive")] == "leren"
    assert note.fields[NOTE_FIELDS.index("Example_1_NL")] == "Ik leer Nederlands op school."
    assert '<section class="variant">' not in "".join(note.fields)


def test_build_note_includes_sound_references(tmp_path: Path) -> None:
    source_item = SourceItem(text="leren", topic="De school", lesson="Lesson 3", exam_level="A2")
    model = create_note_model(DeckSettings())
    audio = NoteAudio(
        word_audio=tmp_path / "word.mp3",
        verb_form_audio=VerbFormAudio(
            infinitive_audio=tmp_path / "verb-infinitive.mp3",
            present_ik_audio=tmp_path / "verb-present-ik.mp3",
            present_hij_audio=tmp_path / "verb-present-hij.mp3",
            past_audio=tmp_path / "verb-past.mp3",
            perfect_audio=tmp_path / "verb-perfect.mp3",
        ),
        example_audios=(
            tmp_path / "present.mp3",
            tmp_path / "past.mp3",
            tmp_path / "perfect.mp3",
        ),
    )

    note = build_note(model, source_item, make_card(), audio=audio)

    assert note.fields[NOTE_FIELDS.index("Word_Audio")] == " [sound:word.mp3]"
    assert note.fields[NOTE_FIELDS.index("Plural_Audio")] == ""
    assert note.fields[NOTE_FIELDS.index("Verb_Infinitive")] == "leren"
    assert note.fields[NOTE_FIELDS.index("Verb_Infinitive_Audio")] == " [sound:verb-infinitive.mp3]"
    assert note.fields[NOTE_FIELDS.index("Verb_Present_Ik")] == "ik leer"
    assert note.fields[NOTE_FIELDS.index("Verb_Present_Ik_Audio")] == " [sound:verb-present-ik.mp3]"
    assert note.fields[NOTE_FIELDS.index("Verb_Present_Hij")] == "hij leert"
    assert note.fields[NOTE_FIELDS.index("Verb_Present_Hij_Audio")] == " [sound:verb-present-hij.mp3]"
    assert note.fields[NOTE_FIELDS.index("Verb_Past")] == "leerde"
    assert note.fields[NOTE_FIELDS.index("Verb_Past_Audio")] == " [sound:verb-past.mp3]"
    assert note.fields[NOTE_FIELDS.index("Verb_Perfect")] == "heeft geleerd"
    assert note.fields[NOTE_FIELDS.index("Verb_Perfect_Audio")] == " [sound:verb-perfect.mp3]"
    assert note.fields[NOTE_FIELDS.index("Example_1_Audio")] == " [sound:present.mp3]"
    assert note.fields[NOTE_FIELDS.index("Example_2_Audio")] == " [sound:past.mp3]"
    assert note.fields[NOTE_FIELDS.index("Example_3_Audio")] == " [sound:perfect.mp3]"


def test_build_note_includes_plural_sound_reference_for_countable_nouns(tmp_path: Path) -> None:
    source_item = SourceItem(text="school", topic="De school", lesson="Lesson 3", exam_level="A2")
    model = create_note_model(DeckSettings())
    card = GeneratedCard(
        dutch_word="de school",
        russian_translation="школа",
        part_of_speech="noun",
        ipa_transcription="sxoːl",
        lesson_topic="De school",
        form_examples=[
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
        tags=["school"],
        plural_form="scholen",
        front_hint="школа (множественное число?)",
    )

    note = build_note(
        model,
        source_item,
        card,
        audio=NoteAudio(plural_audio=tmp_path / "plural.mp3"),
    )

    assert note.fields[NOTE_FIELDS.index("Plural")] == "scholen"
    assert note.fields[NOTE_FIELDS.index("Plural_Audio")] == " [sound:plural.mp3]"


def test_build_example_slot_fields_keeps_each_audio_next_to_matching_sentence(tmp_path: Path) -> None:
    fields = build_example_slot_fields(
        make_card(),
        NoteAudio(
            example_audios=(
                tmp_path / "present.mp3",
                tmp_path / "past.mp3",
                tmp_path / "perfect.mp3",
            )
        ),
    )

    assert fields == [
        "Tegenwoordige tijd: ik leer",
        "Ik leer Nederlands op school.",
        "Я учу нидерландский в школе.",
        " [sound:present.mp3]",
        "Verleden tijd: leerde",
        "Ik leerde gisteren nieuwe woorden.",
        "Вчера я учил новые слова.",
        " [sound:past.mp3]",
        "Perfectum: heeft geleerd",
        "Hij heeft Nederlands geleerd.",
        "Я выучил нидерландский.",
        " [sound:perfect.mp3]",
    ]


def test_build_verb_form_fields_keeps_each_form_editable_with_audio(tmp_path: Path) -> None:
    fields = build_verb_form_fields(
        make_card(),
        NoteAudio(
            verb_form_audio=VerbFormAudio(
                infinitive_audio=tmp_path / "infinitive.mp3",
                present_ik_audio=tmp_path / "present-ik.mp3",
                present_hij_audio=tmp_path / "present-hij.mp3",
                past_audio=tmp_path / "past.mp3",
                perfect_audio=tmp_path / "perfect.mp3",
            )
        ),
    )

    assert fields == [
        "leren",
        " [sound:infinitive.mp3]",
        "ik leer",
        " [sound:present-ik.mp3]",
        "hij leert",
        " [sound:present-hij.mp3]",
        "leerde",
        " [sound:past.mp3]",
        "heeft geleerd",
        " [sound:perfect.mp3]",
        "Notes: regular weak verb",
    ]


def test_format_adjective_forms_only_shows_indeclinable_note() -> None:
    formatted = format_adjective_forms(
        AdjectiveForms(
            onverbuigbaar_example="de gouden ring",
            learner_note="Stofadjectief op -en.",
        )
    )

    assert formatted == "Onverbuigbaar: ja<br>Voorbeeld: de gouden ring<br>Note: Stofadjectief op -en."


def test_format_verb_forms_shows_each_visible_form_on_separate_lines() -> None:
    formatted = format_verb_forms(
        VerbForms(
            infinitive="leren",
            present_ik="ik leer",
            present_hij="hij leert",
            past_tense="leerde",
            perfect_tense="heeft geleerd",
        )
    )

    assert "Tegenwoordige tijd: ik leer<br>hij leert" in formatted
    assert "Verleden tijd: leerde" in formatted
    assert "Perfectum: heeft geleerd" in formatted


def test_format_verb_forms_omits_removed_perfect_example() -> None:
    formatted = format_verb_forms(
        VerbForms(
            infinitive="groeien",
            present_ik="ik groei",
            present_hij="hij groeit",
            past_tense="groeide",
            perfect_tense="is gegroeid",
        )
    )

    assert "Perfectum: is gegroeid" in formatted
    assert "Voltooid deelwoord:" not in formatted
    assert "De kinderen zijn snel gegroeid." not in formatted


def test_deck_package_includes_audio_media(tmp_path: Path) -> None:
    settings = DeckSettings()
    source_item = SourceItem(text="leren", topic="De school", lesson="Lesson 3")
    word_audio = tmp_path / "word.mp3"
    present_audio = tmp_path / "present.mp3"
    past_audio = tmp_path / "past.mp3"
    perfect_audio = tmp_path / "perfect.mp3"
    verb_infinitive_audio = tmp_path / "verb-infinitive.mp3"
    verb_present_ik_audio = tmp_path / "verb-present-ik.mp3"
    verb_present_hij_audio = tmp_path / "verb-present-hij.mp3"
    verb_past_audio = tmp_path / "verb-past.mp3"
    verb_perfect_audio = tmp_path / "verb-perfect.mp3"
    word_audio.write_bytes(b"word")
    present_audio.write_bytes(b"present")
    past_audio.write_bytes(b"past")
    perfect_audio.write_bytes(b"perfect")
    verb_infinitive_audio.write_bytes(b"verb infinitive")
    verb_present_ik_audio.write_bytes(b"verb present ik")
    verb_present_hij_audio.write_bytes(b"verb present hij")
    verb_past_audio.write_bytes(b"verb past")
    verb_perfect_audio.write_bytes(b"verb perfect")
    output_path = tmp_path / "school.apkg"

    build_deck_package(
        [(source_item, make_card())],
        output_path,
        "Lesson 3 - De school",
        settings,
        audio_by_guid={
            build_note_guid(source_item): NoteAudio(
                word_audio=word_audio,
                verb_form_audio=VerbFormAudio(
                    infinitive_audio=verb_infinitive_audio,
                    present_ik_audio=verb_present_ik_audio,
                    present_hij_audio=verb_present_hij_audio,
                    past_audio=verb_past_audio,
                    perfect_audio=verb_perfect_audio,
                ),
                example_audios=(present_audio, past_audio, perfect_audio),
            )
        },
    )

    with zipfile.ZipFile(output_path) as package:
        media = json.loads(package.read("media").decode("utf-8"))

    assert sorted(media.values()) == [
        "past.mp3",
        "perfect.mp3",
        "present.mp3",
        "verb-infinitive.mp3",
        "verb-past.mp3",
        "verb-perfect.mp3",
        "verb-present-hij.mp3",
        "verb-present-ik.mp3",
        "word.mp3",
    ]


def test_deck_package_includes_plural_audio_media(tmp_path: Path) -> None:
    settings = DeckSettings()
    source_item = SourceItem(text="school", topic="De school", lesson="Lesson 3")
    word_audio = tmp_path / "word.mp3"
    plural_audio = tmp_path / "plural.mp3"
    singular_example_audio = tmp_path / "singular-example.mp3"
    plural_example_audio = tmp_path / "plural-example.mp3"
    word_audio.write_bytes(b"word")
    plural_audio.write_bytes(b"plural")
    singular_example_audio.write_bytes(b"singular example")
    plural_example_audio.write_bytes(b"plural example")
    output_path = tmp_path / "school.apkg"

    build_deck_package(
        [(source_item, make_countable_noun_card())],
        output_path,
        "Lesson 3 - De school",
        settings,
        audio_by_guid={
            build_note_guid(source_item): NoteAudio(
                word_audio=word_audio,
                plural_audio=plural_audio,
                example_audios=(singular_example_audio, plural_example_audio),
            )
        },
    )

    with zipfile.ZipFile(output_path) as package:
        media = json.loads(package.read("media").decode("utf-8"))

    assert sorted(media.values()) == [
        "plural-example.mp3",
        "plural.mp3",
        "singular-example.mp3",
        "word.mp3",
    ]


def test_grouped_deck_package_contains_one_note_one_card_and_all_media(
    tmp_path: Path,
) -> None:
    _, source_items, cards = make_clothing_concept_and_cards()
    kleding_word = tmp_path / "kleding.mp3"
    kleding_example = tmp_path / "kleding-example.mp3"
    kleren_word = tmp_path / "kleren.mp3"
    kleren_example = tmp_path / "kleren-example.mp3"
    for media_path in (
        kleding_word,
        kleding_example,
        kleren_word,
        kleren_example,
    ):
        media_path.write_bytes(media_path.name.encode("utf-8"))

    output_path = tmp_path / "clothes.apkg"
    settings = DeckSettings()
    guid = build_note_guid(source_items[0])
    build_deck_package(
        list(zip(source_items, cards, strict=True)),
        output_path,
        "Les 1 - Kleding",
        settings,
        audio_by_guid={
            guid: (
                NoteAudio(
                    word_audio=kleding_word,
                    example_audios=(kleding_example,),
                ),
                NoteAudio(
                    word_audio=kleren_word,
                    example_audios=(kleren_example,),
                ),
            )
        },
    )

    with zipfile.ZipFile(output_path) as package:
        media = json.loads(package.read("media").decode("utf-8"))
        collection_bytes = package.read("collection.anki2")

    connection = sqlite3.connect(":memory:")
    try:
        connection.deserialize(collection_bytes)
        note_rows = connection.execute("select guid, mid, flds from notes").fetchall()
        card_rows = connection.execute("select nid, ord from cards").fetchall()
        model_payload = json.loads(
            connection.execute("select models from col").fetchone()[0]
        )
    finally:
        connection.close()

    assert len(note_rows) == 1
    assert len(card_rows) == 1
    assert card_rows[0][1] == 0
    note_guid, model_id, serialized_fields = note_rows[0]
    assert note_guid == guid
    assert model_id == create_note_model(settings).model_id == 359410166775
    fields = serialized_fields.split("\x1f")
    assert len(fields) == len(NOTE_FIELDS)
    assert fields[NOTE_FIELDS.index("Front")] == "одежда"
    assert fields[NOTE_FIELDS.index("Word_NL")] == "de kleding<br>de kleren"
    assert fields[NOTE_FIELDS.index("POS")].count('<section class="variant">') == 2
    assert [field["name"] for field in model_payload[str(model_id)]["flds"]] == NOTE_FIELDS
    assert sorted(media.values()) == [
        "kleding-example.mp3",
        "kleding.mp3",
        "kleren-example.mp3",
        "kleren.mp3",
    ]


def test_deck_package_write_failure_preserves_existing_output_and_removes_temp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_path = tmp_path / "school.apkg"
    output_path.write_bytes(b"existing complete deck")
    write_paths: list[Path] = []

    def fail_after_partial_write(_package: genanki.Package, filename: str) -> None:
        write_path = Path(filename)
        write_paths.append(write_path)
        write_path.write_bytes(b"partial deck")
        raise RuntimeError("package write failed")

    monkeypatch.setattr(genanki.Package, "write_to_file", fail_after_partial_write)

    with pytest.raises(RuntimeError, match="package write failed"):
        build_deck_package(
            [(SourceItem(text="leren"), make_card())],
            output_path,
            "Lesson 3 - De school",
            DeckSettings(),
        )

    assert len(write_paths) == 1
    assert write_paths[0] != output_path
    assert write_paths[0].parent == output_path.parent
    assert not write_paths[0].exists()
    assert output_path.read_bytes() == b"existing complete deck"
