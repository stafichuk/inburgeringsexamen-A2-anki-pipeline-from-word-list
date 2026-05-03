"""Prompt construction for card generation."""

from __future__ import annotations

import json

from .models import GeneratedCard, SourceItem

PROMPT_VERSION = "2026-05-03.2"


def build_messages(source_item: SourceItem) -> list[dict[str, str]]:
    """Build chat-completion messages for a single vocabulary item."""
    topic = source_item.topic or "Neutral everyday learning context"
    lesson = source_item.lesson or "No lesson title provided"
    exam_level = source_item.exam_level or "A2 Inburgering Spreken"
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

Context:
- topic: {topic}
- lesson: {lesson}
- exam_level: {exam_level}

Rules:
- Infer the part of speech. The user does not provide it manually.
- The card is for active Dutch vocabulary learning for the A2 Inburgering Spreken exam.
- Russian translation must be natural, concise, and learner-friendly.
- Dutch example sentence must be simple A2-level Dutch.
- If a topic is provided, prefer an example sentence that fits that topic.
- If no specific topic is provided, use a neutral everyday-learning context.
- Always fill all common required fields.
- For countable nouns, include the article directly in dutch_word, e.g. "de school" or "het huis"; include plural_form and front_hint. The front_hint must be in Russian and explicitly prompt plural recall, e.g. 'школа (множественное число?)'.
- For uncountable nouns, include the article directly in dutch_word, include front_hint, set plural_form to null, and do not add '(множественное число?)' to front_hint.
- For verbs, include verb_forms with infinitive, present_tense, past_tense, past_participle, and optionally perfect_example, separable_prefix, conjugation_notes.
- For regular adjectives, set adjective_forms to null. Adjective endings are predictable and should not be listed.
- For indeclinable adjectives only, include adjective_forms with onverbuigbaar_example, e.g. "gouden ring", and optionally learner_note. The example must show the adjective in a natural Dutch noun phrase.
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
