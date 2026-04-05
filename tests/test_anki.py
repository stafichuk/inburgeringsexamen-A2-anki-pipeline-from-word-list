from pathlib import Path

from app.anki import NOTE_FIELDS, build_deck_package, create_note_model
from app.config import DeckSettings
from app.models import GeneratedCard, SourceItem, VerbForms


def test_deck_generation_writes_apkg(tmp_path: Path) -> None:
    settings = DeckSettings()
    source_item = SourceItem(text="leren", topic="De school", lesson="Lesson 3", exam_level="A2")
    card = GeneratedCard(
        dutch_word="leren",
        russian_translation="учиться",
        part_of_speech="verb",
        ipa_transcription="ˈleːrə(n)",
        example_sentence_nl="Ik leer Nederlands op school.",
        example_sentence_ru="Я учу нидерландский в школе.",
        lesson_topic="De school",
        tags=["school", "verb"],
        verb_forms=VerbForms(
            infinitive="leren",
            present_tense="ik leer, jij leert, hij leert",
            past_tense="leerde, leerden",
            past_participle="geleerd",
            perfect_example="Ik heb Nederlands geleerd.",
            conjugation_notes="regular weak verb",
        ),
    )
    output_path = tmp_path / "school.apkg"

    build_deck_package([(source_item, card)], output_path, "Lesson 3 - De school", settings)

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_note_model_contains_expected_fields() -> None:
    model = create_note_model(DeckSettings())
    model_field_names = [field["name"] for field in model.fields]
    assert model_field_names == NOTE_FIELDS
