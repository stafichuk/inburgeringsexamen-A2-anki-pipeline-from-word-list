"""Prompt construction for card generation."""

from __future__ import annotations

import json

from .models import GeneratedCard, SourceItem

PROMPT_VERSION = "2026-05-16.2"


def build_messages(source_item: SourceItem) -> list[dict[str, str]]:
    """Build chat-completion messages for a single vocabulary item."""
    topic = source_item.topic or "Neutral everyday learning context"
    lesson = source_item.lesson or "No lesson title provided"
    exam_level = source_item.exam_level or "A2 Inburgering Spreken"
    translation_hint = source_item.translation_hint or "Not provided"
    schema = json.dumps(GeneratedCard.model_json_schema(), ensure_ascii=False, indent=2)

    system_prompt = (
        "You are generating Anki card data for Dutch vocabulary study for Russian-speaking learners. "
        "Return exactly one JSON object and nothing else. "
        "No markdown. No code fences. No commentary."
    )

    user_prompt = f"""
Prompt version: {PROMPT_VERSION}

Task:
Generate one structured JSON object for the Dutch input item below.

Input item:
{source_item.text}

Translation hint:
{translation_hint}

Context:
- topic: {topic}
- lesson: {lesson}
- exam_level: {exam_level}

Rules:
- Infer the part of speech. The user does not provide it manually.
- The card is for active Dutch vocabulary learning for the A2 Inburgering Spreken exam.
- Russian translation must be natural, concise, and learner-friendly.
- If a translation hint is provided, treat it as a strict sense constraint. russian_translation and noun front_hint must use that requested Russian sense, examples must match that meaning, and alternative meanings of the Dutch input must not be merged into this card.
- Never include the translation hint or the ' - ' delimiter in dutch_word.
- Dutch example sentences must be simple A2-level Dutch.
- If a topic is provided, prefer example sentences that fit that topic.
- If no specific topic is provided, use a neutral everyday-learning context.
- Always fill all common required fields.
- Every form_examples entry must use the exact visible Dutch form in the form field, and that form must appear in example_sentence_nl.
- For countable nouns, include the article directly in dutch_word, e.g. "de school" or "het huis"; include plural_form and front_hint. The front_hint must be in Russian and explicitly prompt plural recall, e.g. 'школа (множественное число?)'. Include exactly two form_examples: singular and plural.
- For uncountable nouns, include the article directly in dutch_word, include front_hint, set plural_form to null, and do not add '(множественное число?)' to front_hint. Include exactly one default form_example.
- For verbs, include verb_forms with infinitive, present_tense, past_tense, past_participle, and optionally perfect_example, separable_prefix, conjugation_notes. Set verb_forms.present_tense to the two learner-visible singular present forms: "ik ..." and "hij ...", e.g. "ik leer; hij leert". Include exactly three form_examples: present_tense, past_tense, and past_participle. The form value must be the specific Dutch form used in that example sentence, not a full conjugation table.
- For regular adjectives with two visible forms, set adjective_forms to null and include exactly two form_examples: base_form and e_form. The form values must be the exact adjective forms, e.g. "mooi" and "mooie".
- For adjectives without a distinct -e form, include exactly one single_form example. Choose a sentence where a regular adjective would normally show -e, e.g. "de gouden ring", not an ambiguous context like "een gouden huis". Include adjective_forms only if an exception note is useful.
- For other single-form words, include exactly one default form_example.
- For non-relevant optional fields, use null.
- Keep lesson_topic concise and reflect the lesson/topic context being used for this card.
- tags should be a JSON array of short machine-friendly strings.
- Return valid JSON only.

JSON schema:
{schema}
""".strip()

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
