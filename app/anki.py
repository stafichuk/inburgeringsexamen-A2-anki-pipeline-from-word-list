"""Anki deck generation using genanki."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import html
import os
import re
import tempfile
from pathlib import Path

import genanki

from .config import DeckSettings
from .models import (
    AdjectiveForms,
    FormExample,
    FormExampleKind,
    GeneratedCard,
    SourceConcept,
    SourceItem,
    VerbForms,
    matches_explicit_dutch_answer,
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
    "Verb_Infinitive",
    "Verb_Infinitive_Audio",
    "Verb_Present_Ik",
    "Verb_Present_Ik_Audio",
    "Verb_Present_Hij",
    "Verb_Present_Hij_Audio",
    "Verb_Past",
    "Verb_Past_Audio",
    "Verb_Perfect",
    "Verb_Perfect_Audio",
    "Verb_Notes",
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
.verb-form {
  margin-top: 4px;
}
.verb-label {
  font-weight: bold;
}
.verb-notes {
  margin-top: 6px;
}
.variant + .variant {
  border-top: 1px solid #d9e2ec;
  margin-top: 20px;
  padding-top: 20px;
}
.variant-word {
  font-size: 28px;
  font-weight: bold;
  margin-bottom: 8px;
}
.variant-ipa {
  font-style: italic;
  color: #52606d;
  margin-bottom: 12px;
}
"""

EXAMPLE_KIND_LABELS = {
    FormExampleKind.SINGULAR: "Enkelvoud",
    FormExampleKind.PLURAL: "Meervoud",
    FormExampleKind.DEFAULT: "Voorbeeld",
    FormExampleKind.PRESENT_TENSE: "Tegenwoordige tijd",
    FormExampleKind.PAST_TENSE: "Verleden tijd",
    FormExampleKind.PERFECT_TENSE: "Perfectum",
    FormExampleKind.BASE_FORM: "Zonder -e",
    FormExampleKind.E_FORM: "Met -e",
    FormExampleKind.SINGLE_FORM: "Onverbuigbaar",
}


def stable_anki_id(seed: str) -> int:
    """Convert a seed string into a deterministic positive Anki identifier."""
    digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()
    return int(digest[:10], 16)


@dataclass(frozen=True, slots=True)
class VerbFormAudio:
    """Audio media files attached to individual verb form fields."""

    infinitive_audio: Path | None = None
    present_ik_audio: Path | None = None
    present_hij_audio: Path | None = None
    past_audio: Path | None = None
    perfect_audio: Path | None = None

    def paths(self) -> tuple[Path | None, ...]:
        """Return verb-form audio paths in note-field order."""
        return (
            self.infinitive_audio,
            self.present_ik_audio,
            self.present_hij_audio,
            self.past_audio,
            self.perfect_audio,
        )


@dataclass(frozen=True, slots=True)
class NoteAudio:
    """Audio media files attached to one generated note."""

    word_audio: Path | None = None
    plural_audio: Path | None = None
    verb_form_audio: VerbFormAudio | None = None
    example_audios: tuple[Path | None, ...] = ()


GroupedNoteAudio = tuple[NoteAudio | None, ...]
AudioByGuid = Mapping[str, NoteAudio | GroupedNoteAudio]


def build_note_guid(source_item: SourceItem) -> str:
    """Build the stable note GUID for a source item."""
    concept = source_item.concept
    entry_id = concept.entry_id if concept is not None else source_item.entry_id
    topic = concept.topic if concept is not None else source_item.topic
    lesson = concept.lesson if concept is not None else source_item.lesson
    if entry_id is not None:
        normalized_id = " ".join(entry_id.casefold().split())
        guid_seed = (
            f"id:{normalized_id}|{topic or ''}|{lesson or ''}"
        )
    else:
        identity_text = concept.dutch_answers[0] if concept is not None else source_item.text
        translation_hint = (
            concept.translation_hint if concept is not None else source_item.translation_hint
        )
        normalized_text = " ".join(identity_text.casefold().split())
        normalized_hint = (
            " ".join(translation_hint.casefold().split())
            if translation_hint is not None
            else ""
        )
        guid_seed = (
            f"{normalized_text}|{normalized_hint}|"
            f"{topic or ''}|{lesson or ''}"
        )
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
{{#IPA}}
<div class="word">{{Word_NL}}{{Word_Audio}}{{#Plural}} (meervoud {{Plural}}{{Plural_Audio}}){{/Plural}}</div>
<div class="ipa">{{IPA}}</div>
{{#Verb_Infinitive}}<div class="grammar">
  <div class="label">Werkwoordsvormen:</div>
  <div class="verb-form"><span class="verb-label">Tegenwoordige tijd:</span> {{Verb_Present_Ik}}{{Verb_Present_Ik_Audio}}</div>
  <div class="verb-form">{{Verb_Present_Hij}}{{Verb_Present_Hij_Audio}}</div>
  <div class="verb-form"><span class="verb-label">Verleden tijd:</span> {{Verb_Past}}{{Verb_Past_Audio}}</div>
  <div class="verb-form"><span class="verb-label">Perfectum:</span> {{Verb_Perfect}}{{Verb_Perfect_Audio}}</div>
  {{#Verb_Notes}}<div class="verb-notes">{{Verb_Notes}}</div>{{/Verb_Notes}}
</div>{{/Verb_Infinitive}}
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
{{/IPA}}
{{^IPA}}
{{POS}}
{{#Word_Audio}}<div class="audio">{{Word_Audio}}</div>{{/Word_Audio}}
{{/IPA}}
                """.strip(),
            }
        ],
        css=DEFAULT_CSS,
    )


def _join_html_lines(lines: list[str]) -> str:
    """Join HTML-safe lines with <br> separators."""
    return "<br>".join(html.escape(line) for line in lines if line.strip())


def format_verb_forms(verb_forms: VerbForms | None) -> str:
    """Format verb forms for plain-text display contexts."""
    if verb_forms is None:
        return ""
    lines = [
        f"Infinitive: {verb_forms.infinitive}",
        f"Tegenwoordige tijd: {verb_forms.present_ik}",
        verb_forms.present_hij,
        f"Verleden tijd: {verb_forms.past_tense}",
        f"Perfectum: {verb_forms.perfect_tense}",
    ]
    lines.extend(_verb_note_lines(verb_forms))
    return _join_html_lines(lines)


def _verb_note_lines(verb_forms: VerbForms) -> list[str]:
    """Build optional verb note lines."""
    lines: list[str] = []
    if verb_forms.separable_prefix:
        lines.append(f"Separable prefix: {verb_forms.separable_prefix}")
    if verb_forms.conjugation_notes:
        lines.append(f"Notes: {verb_forms.conjugation_notes}")
    return lines


def format_verb_notes(verb_forms: VerbForms | None) -> str:
    """Format non-form verb notes for the Anki back side."""
    if verb_forms is None:
        return ""
    return _join_html_lines(_verb_note_lines(verb_forms))


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
            hint = _clean_countable_noun_front_hint(
                hint,
                plural_form=card.plural_form,
                fallback=card.russian_translation,
            )
            return html.escape(f"{hint}{PLURAL_PROMPT_SUFFIX}")
        if hint.endswith(PLURAL_PROMPT_SUFFIX):
            hint = hint[: -len(PLURAL_PROMPT_SUFFIX)]
        return html.escape(hint)
    return html.escape(card.russian_translation)


def _clean_countable_noun_front_hint(hint: str, *, plural_form: str, fallback: str) -> str:
    """Keep a Russian cue while removing legacy prompts or leaked Dutch answers."""
    cleaned = hint.strip()
    opening_parenthesis = cleaned.rfind("(")
    if opening_parenthesis >= 0 and cleaned.endswith(")"):
        trailing_group = cleaned[opening_parenthesis:]
        is_plural_question = "мн" in trailing_group.casefold() and "?" in trailing_group
        if is_plural_question or _contains_visible_form(trailing_group, plural_form):
            cleaned = cleaned[:opening_parenthesis].rstrip()

    if _contains_visible_form(cleaned, plural_form):
        cleaned = ""
    return cleaned or fallback.strip()


def _contains_visible_form(text: str, form: str) -> bool:
    """Return whether text contains a form outside a larger word."""
    stripped_form = form.strip()
    if not stripped_form:
        return False
    pattern = rf"(?<!\w){re.escape(stripped_form)}(?!\w)"
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


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


def build_verb_form_fields(card: GeneratedCard, audio: NoteAudio | None = None) -> list[str]:
    """Build editable verb form fields with paired audio fields."""
    if card.verb_forms is None:
        return [""] * 11

    verb_forms = card.verb_forms
    verb_audio = audio.verb_form_audio if audio else None
    return [
        html.escape(verb_forms.infinitive),
        format_audio_reference(verb_audio.infinitive_audio if verb_audio else None),
        html.escape(verb_forms.present_ik),
        format_audio_reference(verb_audio.present_ik_audio if verb_audio else None),
        html.escape(verb_forms.present_hij),
        format_audio_reference(verb_audio.present_hij_audio if verb_audio else None),
        html.escape(verb_forms.past_tense),
        format_audio_reference(verb_audio.past_audio if verb_audio else None),
        html.escape(verb_forms.perfect_tense),
        format_audio_reference(verb_audio.perfect_audio if verb_audio else None),
        format_verb_notes(verb_forms),
    ]


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


def build_variant_html(card: GeneratedCard, audio: NoteAudio | None = None) -> str:
    """Render one complete Dutch answer for a grouped concept note."""
    plural = ""
    if card.plural_form:
        plural_audio = audio.plural_audio if audio else None
        plural = (
            " (meervoud "
            f"{html.escape(card.plural_form)}{format_audio_reference(plural_audio)})"
        )

    parts = [
        '<section class="variant">',
        '<div class="variant-word">'
        f"{html.escape(card.dutch_word)}"
        f"{format_audio_reference(audio.word_audio if audio else None)}"
        f"{plural}</div>",
        f'<div class="variant-ipa">{html.escape(card.ipa_transcription)}</div>',
    ]

    if card.verb_forms is not None:
        verb_forms = card.verb_forms
        verb_audio = audio.verb_form_audio if audio else None
        parts.extend(
            [
                '<div class="grammar">',
                '<div class="label">Werkwoordsvormen:</div>',
                '<div class="verb-form"><span class="verb-label">'
                "Tegenwoordige tijd:</span> "
                f"{html.escape(verb_forms.present_ik)}"
                f"{format_audio_reference(verb_audio.present_ik_audio if verb_audio else None)}"
                "</div>",
                '<div class="verb-form">'
                f"{html.escape(verb_forms.present_hij)}"
                f"{format_audio_reference(verb_audio.present_hij_audio if verb_audio else None)}"
                "</div>",
                '<div class="verb-form"><span class="verb-label">'
                "Verleden tijd:</span> "
                f"{html.escape(verb_forms.past_tense)}"
                f"{format_audio_reference(verb_audio.past_audio if verb_audio else None)}"
                "</div>",
                '<div class="verb-form"><span class="verb-label">Perfectum:</span> '
                f"{html.escape(verb_forms.perfect_tense)}"
                f"{format_audio_reference(verb_audio.perfect_audio if verb_audio else None)}"
                "</div>",
            ]
        )
        verb_notes = format_verb_notes(verb_forms)
        if verb_notes:
            parts.append(f'<div class="verb-notes">{verb_notes}</div>')
        parts.append("</div>")

    adjective_forms = format_adjective_forms(card.adjective_forms)
    if adjective_forms:
        parts.append(
            '<div class="grammar"><span class="label">'
            f"Bijvoeglijk naamwoord:</span><br>{adjective_forms}</div>"
        )

    parts.extend(['<div class="examples">', '<div class="label">Voorbeelden:</div>'])
    example_audios = audio.example_audios if audio else ()
    for index, example in enumerate(card.ordered_form_examples()):
        example_audio = example_audios[index] if index < len(example_audios) else None
        parts.extend(
            [
                '<div class="example">',
                f'<div class="example-form">{format_example_form(example)}</div>',
                f'<div class="example-ru">{html.escape(example.example_sentence_ru)}</div>',
                '<div class="example-nl">'
                f"{html.escape(example.example_sentence_nl)}"
                f"{format_audio_reference(example_audio)}</div>",
                "</div>",
            ]
        )
    parts.extend(["</div>", "</section>"])
    return "".join(parts)


def _ordered_grouped_cards(
    cards: list[tuple[SourceItem, GeneratedCard]],
) -> tuple[SourceConcept, list[tuple[SourceItem, GeneratedCard]]]:
    """Validate and order all generated leaves for one grouped concept."""
    if len(cards) < 2:
        raise ValueError("grouped concepts must contain at least two Dutch answers")

    concept = cards[0][0].concept
    if concept is None:  # pragma: no cover - guarded by the package assembler
        raise ValueError("grouped cards must reference a source concept")

    concept_identity = concept.identity_key()
    if any(
        source_item.concept is None
        or source_item.concept.identity_key() != concept_identity
        for source_item, _ in cards
    ):
        raise ValueError("grouped cards must all reference the same source concept")

    mismatches = [
        (source_item.text, card.dutch_word)
        for source_item, card in cards
        if not matches_explicit_dutch_answer(card.dutch_word, source_item.text)
    ]
    if mismatches:
        requested, generated = mismatches[0]
        raise ValueError(
            "grouped card replaced an explicitly accepted Dutch answer: "
            f"expected {requested!r}, got {generated!r}"
        )

    ordered = sorted(cards, key=lambda item: item[0].answer_index)
    answer_indices = [source_item.answer_index for source_item, _ in ordered]
    expected_indices = list(range(len(concept.dutch_answers)))
    if answer_indices != expected_indices:
        raise ValueError(
            "grouped cards must contain each concept answer exactly once; "
            f"expected answer indexes {expected_indices}, got {answer_indices}"
        )
    return concept, ordered


def build_grouped_note(
    model: genanki.Model,
    cards: list[tuple[SourceItem, GeneratedCard]],
    audio: GroupedNoteAudio | None = None,
) -> genanki.Note:
    """Create one scheduled Anki note containing all accepted Dutch answers."""
    concept, ordered = _ordered_grouped_cards(cards)
    if concept.translation_hint is None:
        raise ValueError("grouped concepts require a shared Russian translation hint")

    if audio is None:
        ordered_audio: GroupedNoteAudio = (None,) * len(ordered)
    else:
        ordered_audio = audio
        if len(ordered_audio) != len(ordered):
            raise ValueError(
                "grouped note audio must contain one entry per Dutch answer; "
                f"expected {len(ordered)}, got {len(ordered_audio)}"
            )

    field_values = dict.fromkeys(NOTE_FIELDS, "")
    field_values.update(
        {
            "Front": html.escape(concept.translation_hint),
            "Word_NL": "<br>".join(
                html.escape(card.dutch_word) for _, card in ordered
            ),
            "Translation_RU": html.escape(concept.translation_hint),
            # A generated single-answer note always has an IPA. Leaving the
            # top-level field empty selects the grouped template branch.
            "IPA": "",
            "POS": '<div class="variants">'
            + "".join(
                build_variant_html(card, variant_audio)
                for (_, card), variant_audio in zip(ordered, ordered_audio, strict=True)
            )
            + "</div>",
            "Lesson": html.escape(concept.lesson or ""),
            "Topic": html.escape(concept.topic or ordered[0][1].lesson_topic),
            "SourceWord": html.escape(concept.source_text()),
        }
    )
    fields = [field_values[field_name] for field_name in NOTE_FIELDS]
    return genanki.Note(
        model=model,
        fields=fields,
        guid=build_note_guid(ordered[0][0]),
    )


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
        *build_verb_form_fields(card, audio),
        format_adjective_forms(card.adjective_forms),
        format_audio_reference(audio.word_audio if audio else None),
        *build_example_slot_fields(card, audio),
        html.escape(source_item.lesson or ""),
        html.escape(source_item.topic or card.lesson_topic),
        html.escape(source_item.text),
    ]
    return genanki.Note(model=model, fields=fields, guid=build_note_guid(source_item))


def _group_cards_by_note(
    cards: list[tuple[SourceItem, GeneratedCard]],
) -> list[list[tuple[SourceItem, GeneratedCard]]]:
    """Group flat variant leaves into learner-facing notes without reordering them."""
    groups: list[list[tuple[SourceItem, GeneratedCard]]] = []
    concept_group_indexes: dict[str, int] = {}
    for source_item, card in cards:
        if source_item.concept is None:
            groups.append([(source_item, card)])
            continue

        concept_identity = source_item.concept.identity_key()
        group_index = concept_group_indexes.get(concept_identity)
        if group_index is None:
            concept_group_indexes[concept_identity] = len(groups)
            groups.append([])
            group_index = len(groups) - 1
        groups[group_index].append((source_item, card))
    return groups


def _iter_audio_paths(card: GeneratedCard, audio: NoteAudio) -> tuple[Path | None, ...]:
    """Return every media path referenced by one rendered Dutch answer."""
    media_paths: list[Path | None] = [audio.word_audio]
    if card.plural_form:
        media_paths.append(audio.plural_audio)
    if card.verb_forms and audio.verb_form_audio:
        media_paths.extend(audio.verb_form_audio.paths())
    media_paths.extend(audio.example_audios)
    return tuple(media_paths)


def build_deck_package(
    cards: list[tuple[SourceItem, GeneratedCard]],
    output_path: Path,
    deck_name: str,
    settings: DeckSettings,
    audio_by_guid: AudioByGuid | None = None,
) -> Path:
    """Write a deck package to disk and return its path."""
    deck_id = stable_anki_id(f"{settings.deck_id_seed}:{deck_name}:deck")
    deck = genanki.Deck(deck_id=deck_id, name=deck_name)
    model = create_note_model(settings)
    media_files: list[str] = []
    seen_media_files: set[Path] = set()

    for note_cards in _group_cards_by_note(cards):
        source_item = note_cards[0][0]
        note_guid = build_note_guid(source_item)
        note_audio = (audio_by_guid or {}).get(note_guid)

        if source_item.concept is None:
            if isinstance(note_audio, tuple):
                raise ValueError("single-answer note audio must be one NoteAudio value")
            deck.add_note(build_note(model, source_item, note_cards[0][1], audio=note_audio))
            variant_audios: GroupedNoteAudio = (note_audio,)
        else:
            if isinstance(note_audio, NoteAudio):
                raise ValueError(
                    "grouped note audio must be a tuple aligned with its Dutch answers"
                )
            grouped_audio = note_audio if note_audio is not None else None
            deck.add_note(build_grouped_note(model, note_cards, audio=grouped_audio))
            _, ordered_cards = _ordered_grouped_cards(note_cards)
            note_cards = ordered_cards
            variant_audios = (
                grouped_audio
                if grouped_audio is not None
                else (None,) * len(note_cards)
            )

        for (_, card), variant_audio in zip(
            note_cards,
            variant_audios,
            strict=True,
        ):
            if variant_audio is None:
                continue
            for media_path in _iter_audio_paths(card, variant_audio):
                if media_path is not None and media_path not in seen_media_files:
                    media_files.append(str(media_path))
                    seen_media_files.add(media_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    package = genanki.Package(deck)
    package.media_files = media_files
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        os.close(file_descriptor)
        package.write_to_file(str(temporary_path))
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return output_path
