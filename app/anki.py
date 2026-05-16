"""Anki deck generation using genanki."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import html
from pathlib import Path
import re

import genanki

from .config import DeckSettings
from .models import (
    AdjectiveForms,
    FormExample,
    FormExampleKind,
    GeneratedCard,
    SourceItem,
    VerbForms,
)

EXAMPLE_SLOT_COUNT = 3

NOTE_FIELDS = [
    "Front",
    "Word_NL",
    "Translation_RU",
    "IPA",
    "POS",
    "Plural",
    "Plural_Audio",
    "Verb_Forms",
    "Adjective_Forms",
    "Word_Audio",
    "Example_1_Form",
    "Example_1_NL",
    "Example_1_RU",
    "Example_1_Audio",
    "Example_2_Form",
    "Example_2_NL",
    "Example_2_RU",
    "Example_2_Audio",
    "Example_3_Form",
    "Example_3_NL",
    "Example_3_RU",
    "Example_3_Audio",
    "Lesson",
    "Topic",
    "SourceWord",
]

PLURAL_PROMPT_SUFFIX = " (множественное число?)"

DEFAULT_CSS = """
.card {
  font-family: Arial, sans-serif;
  font-size: 18px;
  text-align: left;
  color: #1f2933;
  background: #fffdf7;
}
.front {
  font-size: 28px;
  font-weight: bold;
}
.word {
  font-size: 28px;
  font-weight: bold;
  margin-bottom: 8px;
}
.ipa {
  font-style: italic;
  color: #52606d;
  margin-bottom: 12px;
}
.grammar, .meta, .example, .audio {
  margin-top: 12px;
}
.label {
  font-weight: bold;
}
.example-ru {
  margin-bottom: 6px;
}
.example-form {
  font-weight: bold;
  margin-bottom: 4px;
}
"""

EXAMPLE_KIND_LABELS = {
    FormExampleKind.SINGULAR: "Enkelvoud",
    FormExampleKind.PLURAL: "Meervoud",
    FormExampleKind.DEFAULT: "Voorbeeld",
    FormExampleKind.PRESENT_TENSE: "Tegenwoordige tijd",
    FormExampleKind.PAST_TENSE: "Verleden tijd",
    FormExampleKind.PAST_PARTICIPLE: "Voltooid deelwoord",
    FormExampleKind.BASE_FORM: "Zonder -e",
    FormExampleKind.E_FORM: "Met -e",
    FormExampleKind.SINGLE_FORM: "Onverbuigbaar",
}


def stable_anki_id(seed: str) -> int:
    """Convert a seed string into a deterministic positive Anki identifier."""
    digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()
    return int(digest[:10], 16)


@dataclass(frozen=True, slots=True)
class NoteAudio:
    """Audio media files attached to one generated note."""

    word_audio: Path | None = None
    plural_audio: Path | None = None
    example_audios: tuple[Path | None, ...] = ()


def build_note_guid(source_item: SourceItem) -> str:
    """Build the stable note GUID for a source item."""
    guid_seed = f"{source_item.text}|{source_item.topic or ''}|{source_item.lesson or ''}"
    return hashlib.md5(guid_seed.encode("utf-8")).hexdigest()


def create_note_model(settings: DeckSettings) -> genanki.Model:
    """Create the custom note type used by the generated deck."""
    model_id = stable_anki_id(f"{settings.deck_id_seed}:{settings.model_name}:model")
    return genanki.Model(
        model_id=model_id,
        name=settings.model_name,
        fields=[{"name": field_name} for field_name in NOTE_FIELDS],
        templates=[
            {
                "name": "Vocabulary Card",
                "qfmt": '<div class="front">{{Front}}</div>',
                "afmt": """
{{FrontSide}}
<hr id="answer">
<div class="word">{{Word_NL}}{{Word_Audio}}{{#Plural}} (meervoud {{Plural}}{{Plural_Audio}}){{/Plural}}</div>
<div class="ipa">{{IPA}}</div>
{{#Verb_Forms}}<div class="grammar"><span class="label">Werkwoordsvormen:</span><br>{{Verb_Forms}}</div>{{/Verb_Forms}}
{{#Adjective_Forms}}<div class="grammar"><span class="label">Bijvoeglijk naamwoord:</span><br>{{Adjective_Forms}}</div>{{/Adjective_Forms}}
<div class="examples">
  <div class="label">Voorbeelden:</div>
  {{#Example_1_NL}}<div class="example">
    <div class="example-form">{{Example_1_Form}}</div>
    <div class="example-ru">{{Example_1_RU}}</div>
    <div class="example-nl">{{Example_1_NL}}{{Example_1_Audio}}</div>
  </div>{{/Example_1_NL}}
  {{#Example_2_NL}}<div class="example">
    <div class="example-form">{{Example_2_Form}}</div>
    <div class="example-ru">{{Example_2_RU}}</div>
    <div class="example-nl">{{Example_2_NL}}{{Example_2_Audio}}</div>
  </div>{{/Example_2_NL}}
  {{#Example_3_NL}}<div class="example">
    <div class="example-form">{{Example_3_Form}}</div>
    <div class="example-ru">{{Example_3_RU}}</div>
    <div class="example-nl">{{Example_3_NL}}{{Example_3_Audio}}</div>
  </div>{{/Example_3_NL}}
</div>
                """.strip(),
            }
        ],
        css=DEFAULT_CSS,
    )


def _join_html_lines(lines: list[str]) -> str:
    """Join HTML-safe lines with <br> separators."""
    return "<br>".join(html.escape(line) for line in lines if line.strip())


def _split_present_tense_forms(present_tense: str) -> list[str]:
    """Split compact present-tense forms into display lines."""
    return [form.strip() for form in re.split(r"[\n;,]+", present_tense) if form.strip()]


def format_verb_forms(verb_forms: VerbForms | None) -> str:
    """Format verb forms for the Anki back side."""
    if verb_forms is None:
        return ""
    present_tense_lines = _split_present_tense_forms(verb_forms.present_tense)
    lines = [
        f"Infinitive: {verb_forms.infinitive}",
        f"Tegenwoordige tijd: {present_tense_lines[0]}",
        *present_tense_lines[1:],
        f"Verleden tijd: {verb_forms.past_tense}",
        f"Voltooid deelwoord: {verb_forms.past_participle}",
    ]
    if verb_forms.perfect_example:
        lines.append(f"Perfectum: {verb_forms.perfect_example}")
    if verb_forms.separable_prefix:
        lines.append(f"Separable prefix: {verb_forms.separable_prefix}")
    if verb_forms.conjugation_notes:
        lines.append(f"Notes: {verb_forms.conjugation_notes}")
    return _join_html_lines(lines)


def format_adjective_forms(adjective_forms: AdjectiveForms | None) -> str:
    """Format adjective forms for the Anki back side."""
    if adjective_forms is None:
        return ""
    lines = [
        "Onverbuigbaar: ja",
        f"Voorbeeld: {adjective_forms.onverbuigbaar_example}",
    ]
    if adjective_forms.learner_note:
        lines.append(f"Note: {adjective_forms.learner_note}")
    return _join_html_lines(lines)


def build_front(card: GeneratedCard) -> str:
    """Build the front-side Russian prompt."""
    if card.part_of_speech.value == "noun":
        hint = card.front_hint or card.russian_translation
        if card.plural_form:
            return html.escape(card.front_hint or f"{card.russian_translation}{PLURAL_PROMPT_SUFFIX}")
        if hint.endswith(PLURAL_PROMPT_SUFFIX):
            hint = hint[: -len(PLURAL_PROMPT_SUFFIX)]
        return html.escape(hint)
    return html.escape(card.russian_translation)


def format_part_of_speech(card: GeneratedCard) -> str:
    """Convert internal POS values into learner-facing Dutch labels."""
    labels = {
        "noun": "zelfstandig naamwoord",
        "verb": "werkwoord",
        "adjective": "bijvoeglijk naamwoord",
        "adverb": "bijwoord",
        "pronoun": "voornaamwoord",
        "preposition": "voorzetsel",
        "conjunction": "voegwoord",
        "phrase": "woordgroep",
        "expression": "uitdrukking",
        "other": "overig",
    }
    return html.escape(labels.get(card.part_of_speech.value, card.part_of_speech.value))


def format_audio_reference(path: Path | None) -> str:
    """Format an Anki sound reference for a packaged media file."""
    if path is None:
        return ""
    return f" [sound:{html.escape(path.name)}]"


def format_example_form(example: FormExample) -> str:
    """Format a form label for the Anki back side."""
    label = EXAMPLE_KIND_LABELS[example.kind]
    return html.escape(f"{label}: {example.form}")


def build_example_slot_fields(card: GeneratedCard, audio: NoteAudio | None = None) -> list[str]:
    """Build the fixed example slot fields used by the note model."""
    examples = card.ordered_form_examples()
    example_audios = audio.example_audios if audio else ()
    fields: list[str] = []

    for index in range(EXAMPLE_SLOT_COUNT):
        if index >= len(examples):
            fields.extend(["", "", "", ""])
            continue

        example = examples[index]
        example_audio = example_audios[index] if index < len(example_audios) else None
        fields.extend(
            [
                format_example_form(example),
                html.escape(example.example_sentence_nl),
                html.escape(example.example_sentence_ru),
                format_audio_reference(example_audio),
            ]
        )

    return fields


def build_note(
    model: genanki.Model,
    source_item: SourceItem,
    card: GeneratedCard,
    audio: NoteAudio | None = None,
) -> genanki.Note:
    """Create a genanki note from a validated card."""
    fields = [
        build_front(card),
        html.escape(card.dutch_word),
        html.escape(card.russian_translation),
        html.escape(card.ipa_transcription),
        format_part_of_speech(card),
        html.escape(card.plural_form or ""),
        format_audio_reference(audio.plural_audio if audio and card.plural_form else None),
        format_verb_forms(card.verb_forms),
        format_adjective_forms(card.adjective_forms),
        format_audio_reference(audio.word_audio if audio else None),
        *build_example_slot_fields(card, audio),
        html.escape(source_item.lesson or ""),
        html.escape(source_item.topic or card.lesson_topic),
        html.escape(source_item.text),
    ]
    return genanki.Note(model=model, fields=fields, guid=build_note_guid(source_item))


def build_deck_package(
    cards: list[tuple[SourceItem, GeneratedCard]],
    output_path: Path,
    deck_name: str,
    settings: DeckSettings,
    audio_by_guid: Mapping[str, NoteAudio] | None = None,
) -> Path:
    """Write a deck package to disk and return its path."""
    deck_id = stable_anki_id(f"{settings.deck_id_seed}:{deck_name}:deck")
    deck = genanki.Deck(deck_id=deck_id, name=deck_name)
    model = create_note_model(settings)
    media_files: list[str] = []
    seen_media_files: set[Path] = set()

    for source_item, card in cards:
        audio = (audio_by_guid or {}).get(build_note_guid(source_item))
        deck.add_note(build_note(model, source_item, card, audio=audio))
        if audio is not None:
            media_paths = [audio.word_audio]
            if card.plural_form:
                media_paths.append(audio.plural_audio)
            media_paths.extend(audio.example_audios)
            for media_path in media_paths:
                if media_path is not None and media_path not in seen_media_files:
                    media_files.append(str(media_path))
                    seen_media_files.add(media_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    package = genanki.Package(deck)
    package.media_files = media_files
    package.write_to_file(str(output_path))
    return output_path
