"""Prompt construction for coordinated card generation."""

from __future__ import annotations

import json

from .models import GeneratedCard, SourceItem

PROMPT_VERSION = "2026-07-17.1"


def _pending_payload(pending_items: list[tuple[int, SourceItem]]) -> list[dict[str, object]]:
    """Build the compact request payload shown to the model."""
    return [
        {
            "source_id": source_id,
            "input_item": source_item.text,
            "translation_hint": source_item.translation_hint,
            "topic": source_item.topic,
            "lesson": source_item.lesson,
            "exam_level": source_item.exam_level,
        }
        for source_id, source_item in pending_items
    ]


def _existing_payload(
    existing_cards: list[tuple[SourceItem, GeneratedCard]],
) -> list[dict[str, object]]:
    """Build Dutch-only immutable context from accepted cards."""
    return [
        {
            "input_item": source_item.text,
            "dutch_word": card.dutch_word,
            "examples": [
                {
                    "form": example.form,
                    "sentence_nl": example.example_sentence_nl,
                }
                for example in card.ordered_form_examples()
            ],
        }
        for source_item, card in existing_cards
    ]


def build_messages(
    pending_items: list[tuple[int, SourceItem]],
    existing_cards: list[tuple[SourceItem, GeneratedCard]],
    *,
    validation_error: str | None = None,
) -> list[dict[str, str]]:
    """Build messages for one coordinated batch of unresolved vocabulary items."""
    if not pending_items:
        raise ValueError("pending_items must not be empty")

    pending_json = json.dumps(_pending_payload(pending_items), ensure_ascii=False, indent=2)
    existing_json = json.dumps(_existing_payload(existing_cards), ensure_ascii=False, indent=2)
    card_schema = json.dumps(GeneratedCard.model_json_schema(), ensure_ascii=False, indent=2)

    system_prompt = (
        "You are generating Anki card data for Dutch vocabulary study for Russian-speaking learners. "
        "Return exactly one JSON object and nothing else. "
        "No markdown. No code fences. No commentary."
    )

    user_prompt = f"""
Prompt version: {PROMPT_VERSION}

Task:
Generate cards for every unresolved input item as one coordinated batch.

Unresolved input items:
{pending_json}

Previously accepted cards (immutable diversity context; do not return or modify these cards):
{existing_json}

Global diversity rules:
- Treat the lesson topic as metadata and a soft theme, not as a mandatory setting for every example.
- Prefer the target word's most natural, common A2 usage, even when that usage is outside the lesson topic.
- Mention words from the topic only when they add useful meaning; never insert them merely to appear relevant.
- Consider the accepted examples and all examples in this response as one corpus.
- Choose contexts yourself and distribute them naturally across people, places, actions, questions, descriptions, requests, plans, and past events.
- Do not reuse the same target-masked sentence frame for different vocabulary items.
- Avoid repeatedly using the same subject-verb-object pattern or the same topic nouns.
- Examples for different forms of one card should use different situations or sentence structures when natural.
- Correct meaning, natural Dutch, and A2 clarity take priority over novelty.

Card rules:
- Infer the part of speech. The user does not provide it manually.
- The cards are for active Dutch vocabulary learning for the A2 Inburgering Spreken exam.
- Russian translations must be natural, concise, and learner-friendly.
- If a translation hint is provided, treat it as a strict sense constraint. russian_translation and noun front_hint must use that requested Russian sense, examples must match that meaning, and alternative meanings must not be merged into the card.
- Never include the translation hint or the ' - ' delimiter in dutch_word.
- Dutch example sentences must be simple A2-level Dutch.
- Always fill all common required fields.
- For non-verbs, every form_examples entry must use the exact visible Dutch form in the form field, and that form must appear in example_sentence_nl.
- For nouns, dutch_word must include the article directly, e.g. "de tante" or "het huis". In noun form_examples, form must be the bare noun that is visible in the sentence. Never include de or het in a noun form_examples form. For example, dutch_word "het hoofd" uses singular form "hoofd" in "Mijn hoofd doet pijn.", and dutch_word "de tante" uses singular form "tante" in "Mijn tante woont in Amsterdam.".
- Month names are Dutch de-nouns. For month-name cards, set dutch_word to "de januari", "de februari", "de maart", "de april", "de mei", "de juni", "de juli", "de augustus", "de september", "de oktober", "de november", or "de december". Use the bare month name in form_examples. Set plural_form to null and include one default form_example unless the input explicitly asks for a plural month form.
- If an input item is already a plural noun, normalize it to the singular lemma in dutch_word and keep the input plural as plural_form. For example, input "de ouders" must produce dutch_word "de ouder" and plural_form "ouders".
- For countable nouns, plural_form must be the bare plural form without article. Include front_hint in Russian and explicitly prompt plural recall. Include exactly two form_examples: singular and plural.
- For uncountable nouns, include front_hint, set plural_form to null, do not prompt plural recall, and include exactly one default form_example.
- For verbs, include verb_forms with infinitive, present_ik, present_hij, past_tense, perfect_tense, and optionally separable_prefix and conjugation_notes. Include exactly three form_examples: present_tense, past_tense, and perfect_tense.
- For regular adjectives with two visible forms, set adjective_forms to null and include exactly two form_examples: base_form and e_form.
- For adjectives without a distinct -e form, include exactly one single_form example in a context where a regular adjective would normally show -e. Include adjective_forms only if an exception note is useful.
- For other single-form words, include exactly one default form_example.
- For non-relevant optional fields, use null.
- Keep lesson_topic concise and reflect the lesson/topic metadata for the card.
- tags must be a JSON array of short machine-friendly strings.

Output contract:
- Return one object with exactly one top-level field named "cards".
- cards must contain exactly one object for every source_id in the unresolved input list.
- Do not return accepted reference cards or invent source IDs.
- Every item must echo source_id, input_item, and translation_hint exactly as they appear together in the unresolved input list.
- Do not normalize, correct, translate, or otherwise change the echoed input_item or translation_hint.
- Always include translation_hint in the wrapper; echo it as JSON null when it is null in the input list.
- Each item must have this shape: {{"source_id": 1, "input_item": "exact original input_item", "translation_hint": null, "card": {{...}}}}.
- The nested card object must match the schema below.

GeneratedCard JSON schema:
{card_schema}
""".strip()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    if validation_error is not None:
        repair_prompt = f"""
The previous batch attempt left some requested source IDs unresolved.
The unresolved input list above already contains only the cards that still need to be generated.
Return a corrected JSON object for every listed source_id and no other cards.
Echo the exact input_item and translation_hint paired with each source_id in every returned wrapper.
Include translation_hint explicitly as JSON null when the unresolved input item has no hint.

Validation feedback:
{validation_error}
""".strip()
        messages.append({"role": "user", "content": repair_prompt})

    return messages
