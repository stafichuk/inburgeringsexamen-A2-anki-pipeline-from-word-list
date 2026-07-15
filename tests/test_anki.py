import hashlib
import json
from pathlib import Path
import zipfile

import genanki
import pytest

from app.anki import (
    NOTE_FIELDS,
    NoteAudio,
    VerbFormAudio,
    build_deck_package,
    build_example_slot_fields,
    build_front,
    build_note,
    build_note_guid,
    build_verb_form_fields,
    create_note_model,
    format_adjective_forms,
    format_verb_forms,
)
from app.config import DeckSettings
from app.models import AdjectiveForms, GeneratedCard, SourceItem, VerbForms


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
    assert "{{POS}}" not in template
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
