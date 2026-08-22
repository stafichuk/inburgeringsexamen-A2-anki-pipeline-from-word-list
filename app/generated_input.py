"""Validated input format for assembling decks from pre-generated card data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import Field, ValidationError, field_validator

from .models import (
    GeneratedCard,
    SourceConcept,
    SourceItem,
    StrictModel,
    matches_explicit_dutch_answer,
)


class GeneratedAnswerInput(StrictModel):
    """One authored Dutch answer paired with its pre-generated card data."""

    input_item: str
    card: GeneratedCard

    @field_validator("input_item")
    @classmethod
    def ensure_input_item(cls, value: str) -> str:
        """Reject blank source answers."""
        if not value.strip():
            raise ValueError("input_item must not be empty")
        return value


class GeneratedConceptInput(StrictModel):
    """One learner-facing concept with one or more generated Dutch answers."""

    entry_id: str | None = None
    translation_hint: str | None = None
    topic: str | None = None
    lesson: str | None = None
    exam_level: str | None = None
    answers: tuple[GeneratedAnswerInput, ...] = Field(min_length=1)


class GeneratedDataManifest(StrictModel):
    """Versioned, ordered bundle consumed by generated-data pipeline mode."""

    format: Literal["dutch-a2-generated-cards"]
    schema_version: Literal[1]
    concepts: tuple[GeneratedConceptInput, ...] = Field(min_length=1)


def _resolve_context_value(
    override: str | None,
    embedded: str | None,
    default: str | None,
) -> str | None:
    """Apply an explicit CLI override, then embedded metadata, then config default."""
    if override is not None:
        return override
    if embedded is not None:
        return embedded
    return default


def load_generated_cards(
    input_path: Path,
    *,
    topic: str | None = None,
    lesson: str | None = None,
    exam_level: str | None = None,
    default_topic: str | None = None,
    default_lesson: str | None = None,
    default_exam_level: str | None = None,
) -> list[tuple[SourceItem, GeneratedCard]]:
    """Load and validate a complete generated-data manifest in authored order."""
    if not input_path.exists():
        raise FileNotFoundError(f"generated input file not found: {input_path}")

    try:
        payload = json.loads(input_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid generated input JSON: {exc}") from exc

    try:
        manifest = GeneratedDataManifest.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"invalid generated input: {exc}") from exc

    cards: list[tuple[SourceItem, GeneratedCard]] = []
    concept_positions: dict[str, int] = {}
    source_positions: dict[str, int] = {}
    for position, generated_concept in enumerate(manifest.concepts, start=1):
        source_concept = SourceConcept(
            entry_id=generated_concept.entry_id,
            dutch_answers=tuple(answer.input_item for answer in generated_concept.answers),
            translation_hint=generated_concept.translation_hint,
            topic=_resolve_context_value(
                topic,
                generated_concept.topic,
                default_topic,
            ),
            lesson=_resolve_context_value(
                lesson,
                generated_concept.lesson,
                default_lesson,
            ),
            exam_level=_resolve_context_value(
                exam_level,
                generated_concept.exam_level,
                default_exam_level,
            ),
        )

        concept_identity = source_concept.identity_key()
        if concept_identity in concept_positions:
            raise ValueError(
                "duplicate source identity in concepts "
                f"{concept_positions[concept_identity]} and {position}; "
                "use distinct entry_id values"
            )
        concept_positions[concept_identity] = position

        source_items = source_concept.source_items()
        for source_item, generated_answer in zip(
            source_items,
            generated_concept.answers,
            strict=True,
        ):
            source_identity = source_item.identity_key()
            if source_identity in source_positions:
                raise ValueError(
                    "duplicate source identity in concepts "
                    f"{source_positions[source_identity]} and {position}; "
                    "use distinct entry_id values"
                )
            source_positions[source_identity] = position

            if not matches_explicit_dutch_answer(
                generated_answer.card.dutch_word,
                source_item.text,
            ):
                raise ValueError(
                    f"concept {position} answer {source_item.answer_index + 1} replaced "
                    "an explicitly accepted Dutch answer: "
                    f"expected {source_item.text!r}, got {generated_answer.card.dutch_word!r}"
                )
            cards.append((source_item, generated_answer.card))

    return cards
