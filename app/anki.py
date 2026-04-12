"""Anki deck generation using genanki."""

from __future__ import annotations

import hashlib
import html
from pathlib import Path

import genanki

from .config import DeckSettings
from .models import AdjectiveForms, GeneratedCard, SourceItem, VerbForms

NOTE_FIELDS = [
    "Front",
    "Word_NL",
    "Translation_RU",
    "IPA",
    "POS",
    "Article",
    "Plural",
    "Verb_Forms",
    "Adjective_Forms",
    "Example_NL",
    "Example_RU",
    "Word_Audio",
    "Example_Audio",
    "Lesson",
    "Topic",
    "SourceWord",
]

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
"""


def stable_anki_id(seed: str) -> int:
    """Convert a seed string into a deterministic positive Anki identifier."""
    digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()
    return int(digest[:10], 16)


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
<div class="word">{{#Article}}{{Article}} {{/Article}}{{Word_NL}}{{#Plural}} (meervoud {{Plural}}){{/Plural}}</div>
<div class="ipa">{{IPA}}</div>
{{#Verb_Forms}}<div class="grammar"><span class="label">Werkwoordsvormen:</span><br>{{Verb_Forms}}</div>{{/Verb_Forms}}
{{#Adjective_Forms}}<div class="grammar"><span class="label">Bijvoeglijke vormen:</span><br>{{Adjective_Forms}}</div>{{/Adjective_Forms}}
<div class="example">
  <div class="label">Voorbeeld:</div>
  <div class="example-ru">{{Example_RU}}</div>
  <div class="example-nl">{{Example_NL}}</div>
</div>
<div class="meta"><span class="label">Woordsoort:</span> {{POS}}</div>
{{#Word_Audio}}<div class="audio">{{Word_Audio}}</div>{{/Word_Audio}}
{{#Example_Audio}}<div class="audio">{{Example_Audio}}</div>{{/Example_Audio}}
                """.strip(),
            }
        ],
        css=DEFAULT_CSS,
    )


def _join_html_lines(lines: list[str]) -> str:
    """Join HTML-safe lines with <br> separators."""
    return "<br>".join(html.escape(line) for line in lines if line.strip())


def format_verb_forms(verb_forms: VerbForms | None) -> str:
    """Format verb forms for the Anki back side."""
    if verb_forms is None:
        return ""
    lines = [
        f"Infinitive: {verb_forms.infinitive}",
        f"Tegenwoordige tijd: {verb_forms.present_tense}",
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
        f"Base form: {adjective_forms.base_form}",
        f"de-word form: {adjective_forms.de_form}",
        f"het-word form: {adjective_forms.het_form}",
    ]
    if adjective_forms.learner_note:
        lines.append(f"Note: {adjective_forms.learner_note}")
    return _join_html_lines(lines)


def build_front(card: GeneratedCard) -> str:
    """Build the front-side Russian prompt."""
    if card.part_of_speech.value == "noun":
        return html.escape(card.front_hint or f"{card.russian_translation} (множественное число?)")
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


def build_note(model: genanki.Model, source_item: SourceItem, card: GeneratedCard) -> genanki.Note:
    """Create a genanki note from a validated card."""
    fields = [
        build_front(card),
        html.escape(card.dutch_word),
        html.escape(card.russian_translation),
        html.escape(card.ipa_transcription),
        format_part_of_speech(card),
        html.escape(card.article or ""),
        html.escape(card.plural_form or ""),
        format_verb_forms(card.verb_forms),
        format_adjective_forms(card.adjective_forms),
        html.escape(card.example_sentence_nl),
        html.escape(card.example_sentence_ru),
        "",
        "",
        html.escape(source_item.lesson or ""),
        html.escape(source_item.topic or card.lesson_topic),
        html.escape(source_item.text),
    ]
    guid_seed = f"{source_item.text}|{source_item.topic or ''}|{source_item.lesson or ''}"
    return genanki.Note(model=model, fields=fields, guid=hashlib.md5(guid_seed.encode("utf-8")).hexdigest())


def build_deck_package(
    cards: list[tuple[SourceItem, GeneratedCard]],
    output_path: Path,
    deck_name: str,
    settings: DeckSettings,
) -> Path:
    """Write a deck package to disk and return its path."""
    deck_id = stable_anki_id(f"{settings.deck_id_seed}:{deck_name}:deck")
    deck = genanki.Deck(deck_id=deck_id, name=deck_name)
    model = create_note_model(settings)

    for source_item, card in cards:
        deck.add_note(build_note(model, source_item, card))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    package = genanki.Package(deck)
    package.write_to_file(str(output_path))
    return output_path
