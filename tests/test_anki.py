import json
from pathlib import Path
import zipfile

from app.anki import (
    NOTE_FIELDS,
    NoteAudio,
    build_front,
    build_deck_package,
    build_note,
    build_note_guid,
    create_note_model,
    format_adjective_forms,
)
from app.config import DeckSettings
from app.models import AdjectiveForms, GeneratedCard, SourceItem, VerbForms


def make_card(word: str = "leren") -> GeneratedCard:
    return GeneratedCard(
        dutch_word=word,
        russian_translation="учиться",
        part_of_speech="verb",
        ipa_transcription="ˈleːrə(n)",
        example_sentence_nl="Ik leer Nederlands op school.",
        example_sentence_ru="Я учу нидерландский в школе.",
        lesson_topic="De school",
        tags=["school", "verb"],
        verb_forms=VerbForms(
            infinitive=word,
            present_tense=f"ik {word}, jij {word}t, hij {word}t",
            past_tense="leerde, leerden",
            past_participle="geleerd",
            perfect_example="Ik heb Nederlands geleerd.",
            conjugation_notes="regular weak verb",
        ),
    )


def test_deck_generation_writes_apkg(tmp_path: Path) -> None:
    settings = DeckSettings()
    source_item = SourceItem(text="leren", topic="De school", lesson="Lesson 3", exam_level="A2")
    card = make_card()
    output_path = tmp_path / "school.apkg"

    build_deck_package([(source_item, card)], output_path, "Lesson 3 - De school", settings)

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_note_model_contains_expected_fields() -> None:
    model = create_note_model(DeckSettings())
    model_field_names = [field["name"] for field in model.fields]
    assert model_field_names == NOTE_FIELDS


def test_note_model_template_matches_updated_layout() -> None:
    model = create_note_model(DeckSettings())
    template = model.templates[0]["afmt"]
    css = model.css

    assert "Woordsoort:" not in template
    assert "{{POS}}" not in template
    assert "{{Article}}" not in template
    assert "Voorbeeld:" in template
    assert "Werkwoordsvormen:" in template
    assert "Bijvoeglijk naamwoord:" in template
    assert "Lesson:" not in template
    assert "Topic:" not in template
    assert "Article:" not in template
    assert "(meervoud {{Plural}})" in template
    assert "color: #6b1d1d;" not in css


def test_build_front_does_not_add_plural_prompt_for_uncountable_noun() -> None:
    card = GeneratedCard(
        dutch_word="de melk",
        russian_translation="молоко",
        part_of_speech="noun",
        ipa_transcription="mɛlk",
        example_sentence_nl="Ik drink melk.",
        example_sentence_ru="Я пью молоко.",
        lesson_topic="Eten en drinken",
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
        example_sentence_nl="Mijn school is dichtbij.",
        example_sentence_ru="Моя школа находится рядом.",
        lesson_topic="De school",
        tags=["school"],
        plural_form="scholen",
        front_hint="школа (множественное число?)",
    )

    note = build_note(model, source_item, card)

    assert "Article" not in NOTE_FIELDS
    assert note.fields[1] == "de school"


def test_build_note_includes_sound_references(tmp_path: Path) -> None:
    source_item = SourceItem(text="leren", topic="De school", lesson="Lesson 3", exam_level="A2")
    model = create_note_model(DeckSettings())
    audio = NoteAudio(
        word_audio=tmp_path / "word.mp3",
        example_audio=tmp_path / "example.mp3",
    )

    note = build_note(model, source_item, make_card(), audio=audio)

    assert note.fields[10] == " [sound:word.mp3]"
    assert note.fields[11] == " [sound:example.mp3]"


def test_format_adjective_forms_only_shows_indeclinable_note() -> None:
    formatted = format_adjective_forms(
        AdjectiveForms(
            onverbuigbaar_example="gouden ring",
            learner_note="Stofadjectief op -en.",
        )
    )

    assert formatted == "Onverbuigbaar: ja<br>Voorbeeld: gouden ring<br>Note: Stofadjectief op -en."


def test_deck_package_includes_audio_media(tmp_path: Path) -> None:
    settings = DeckSettings()
    source_item = SourceItem(text="leren", topic="De school", lesson="Lesson 3", exam_level="A2")
    word_audio = tmp_path / "word.mp3"
    example_audio = tmp_path / "example.mp3"
    word_audio.write_bytes(b"word")
    example_audio.write_bytes(b"example")
    output_path = tmp_path / "school.apkg"

    build_deck_package(
        [(source_item, make_card())],
        output_path,
        "Lesson 3 - De school",
        settings,
        audio_by_guid={
            build_note_guid(source_item): NoteAudio(
                word_audio=word_audio,
                example_audio=example_audio,
            )
        },
    )

    with zipfile.ZipFile(output_path) as package:
        media = json.loads(package.read("media").decode("utf-8"))

    assert sorted(media.values()) == ["example.mp3", "word.mp3"]
