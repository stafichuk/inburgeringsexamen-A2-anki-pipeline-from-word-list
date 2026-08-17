"""End-to-end generation pipeline."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from .anki import NoteAudio, VerbFormAudio, build_deck_package, build_note_guid
from .audio import AudioGenerationError, AudioGenerator, build_audio_generator
from .cache import CardCache
from .config import AppSettings
from .llm_client import BatchGenerationResult, LLMClient, LLMClientError
from .models import GeneratedCard, SourceConcept, SourceItem, matches_explicit_dutch_answer
from .prompts import PROMPT_VERSION

LOGGER = logging.getLogger(__name__)
SOURCE_TRANSLATION_DELIMITER = " - "
SOURCE_ALTERNATIVE_DELIMITER = " | "
EXPLICIT_ID_PATTERN = re.compile(r"^\[([^\[\]]+)\]\s+(.+)$")


class CardGenerator(Protocol):
    """Protocol for card generation backends."""

    def generate_cards(
        self,
        pending_items: list[tuple[int, SourceItem]],
        existing_cards: list[tuple[SourceItem, GeneratedCard]],
        on_card_accepted: Callable[[int, GeneratedCard], None] | None = None,
    ) -> BatchGenerationResult:
        """Generate a coordinated batch while preserving partial successes."""


@dataclass(slots=True)
class FailedItem:
    """An unrecoverable item failure recorded during a pipeline run."""

    source_word: str
    reason: str


@dataclass(slots=True)
class AudioFailedItem:
    """An unrecoverable audio failure recorded during a pipeline run."""

    source_word: str
    field_name: str
    reason: str


@dataclass(slots=True)
class PipelineResult:
    """Summary of a pipeline run."""

    output_path: Path
    total_items: int
    generated_items: int
    cached_items: int
    deck_written: bool = True
    failed_items: list[FailedItem] = field(default_factory=list)
    audio_failed_items: list[AudioFailedItem] = field(default_factory=list)


def _group_source_indices(source_items: list[SourceItem]) -> list[list[int]]:
    """Group one-based answer indices by their learner-facing concept."""
    groups: list[list[int]] = []
    group_positions: dict[str, int] = {}
    for index, source_item in enumerate(source_items, start=1):
        identity = source_item.concept_identity_key()
        position = group_positions.get(identity)
        if position is None:
            group_positions[identity] = len(groups)
            groups.append([index])
        else:
            groups[position].append(index)
    return groups


def _count_complete_groups(groups: list[list[int]], available_indices: set[int]) -> int:
    """Count concepts for which every accepted Dutch answer is available."""
    return sum(all(index in available_indices for index in group) for group in groups)


@dataclass(slots=True)
class DeckGenerationPipeline:
    """Coordinate source loading, generation, caching, and deck writing."""

    settings: AppSettings
    llm_client: CardGenerator | None = None
    audio_generator: AudioGenerator | None = None
    cache: CardCache | None = None

    def __post_init__(self) -> None:
        if self.llm_client is None:
            self.llm_client = LLMClient(
                self.settings.llm,
                json_repair_enabled=self.settings.generation.json_repair_enabled,
            )
        if self.cache is None:
            self.cache = CardCache(self.settings.cache.directory)
        if self.settings.audio.enabled and self.audio_generator is None:
            self.audio_generator = build_audio_generator(self.settings.audio)

    def run(
        self,
        *,
        input_path: Path,
        output_path: Path,
        topic: str | None = None,
        lesson: str | None = None,
        exam_level: str | None = None,
        force: bool = False,
    ) -> PipelineResult:
        """Run the full generation pipeline for a word list."""
        source_items = load_source_items(
            input_path,
            topic=topic or self.settings.generation.default_topic,
            lesson=lesson or self.settings.generation.default_lesson,
            exam_level=exam_level or self.settings.generation.default_exam_level,
        )
        source_groups = _group_source_indices(source_items)
        cards_by_index: dict[int, tuple[SourceItem, GeneratedCard]] = {}
        failed_items: list[FailedItem] = []
        cached_indices: set[int] = set()
        pending_items: list[tuple[int, SourceItem]] = []

        if force:
            self.cache.begin_refresh(source_items)

        for index, source_item in enumerate(source_items, start=1):
            LOGGER.info("Preparing %s/%s: %s", index, len(source_items), source_item.text)
            try:
                if force:
                    card = None
                else:
                    card = self.cache.get(source_item)
                if (
                    card is not None
                    and source_item.concept is not None
                    and not matches_explicit_dutch_answer(card.dutch_word, source_item.text)
                ):
                    LOGGER.warning(
                        "Ignoring cached card for grouped answer '%s' because it contains '%s'.",
                        source_item.text,
                        card.dutch_word,
                    )
                    card = None
                if card is not None:
                    cached_indices.add(index)
                    LOGGER.info("Cache hit for '%s'.", source_item.text)
                    cards_by_index[index] = (source_item, card)
                else:
                    pending_items.append((index, source_item))
            except (LLMClientError, ValueError) as exc:
                LOGGER.error("Failed to process '%s': %s", source_item.text, exc)
                failed_items.append(FailedItem(source_word=source_item.text, reason=str(exc)))

        self._generate_pending_items(
            pending_items=pending_items,
            total_items=len(source_items),
            cards_by_index=cards_by_index,
            failed_items=failed_items,
        )

        cards = [cards_by_index[index] for index in sorted(cards_by_index)]
        accepted_indices = set(cards_by_index)
        cached_items = _count_complete_groups(source_groups, cached_indices)
        generated_items = _count_complete_groups(source_groups, accepted_indices)
        if failed_items:
            LOGGER.error(
                "Deck was not written because %s/%s answer(s) remain unresolved.",
                len(failed_items),
                len(source_items),
            )
            return PipelineResult(
                output_path=output_path,
                total_items=len(source_groups),
                generated_items=generated_items,
                cached_items=cached_items,
                deck_written=False,
                failed_items=failed_items,
            )

        if not cards:  # pragma: no cover - guarded by non-empty input and failures above
            raise RuntimeError("no valid cards were generated; deck will not be written")

        audio_failed_items: list[AudioFailedItem] = []
        audio_by_guid = self._generate_audio_for_cards(cards, audio_failed_items)

        deck_name = resolve_deck_name(
            output_path=output_path,
            settings=self.settings,
            topic=topic,
            lesson=lesson,
        )
        build_deck_package(cards, output_path, deck_name, self.settings.deck, audio_by_guid=audio_by_guid)
        return PipelineResult(
            output_path=output_path,
            total_items=len(source_groups),
            generated_items=generated_items,
            cached_items=cached_items,
            failed_items=failed_items,
            audio_failed_items=audio_failed_items,
        )

    def _generate_audio_for_cards(
        self,
        cards: list[tuple[SourceItem, GeneratedCard]],
        audio_failed_items: list[AudioFailedItem],
    ) -> dict[str, NoteAudio | tuple[NoteAudio | None, ...]]:
        """Generate per-answer audio and group it by the learner-facing note GUID."""
        if not self.settings.audio.enabled or self.audio_generator is None:
            return {}

        audio_by_guid: dict[str, NoteAudio | tuple[NoteAudio | None, ...]] = {}
        grouped_audio: dict[str, list[NoteAudio | None]] = {}
        for source_item, card in cards:
            LOGGER.info("Generating audio for '%s'.", source_item.text)
            word_audio = self._generate_one_audio(
                source_item=source_item,
                text=card.dutch_word,
                field_name="Word_Audio",
                label="word",
                audio_failed_items=audio_failed_items,
            )
            plural_audio = (
                self._generate_one_audio(
                    source_item=source_item,
                    text=card.plural_form,
                    field_name="Plural_Audio",
                    label="plural",
                    audio_failed_items=audio_failed_items,
                )
                if card.plural_form
                else None
            )
            verb_form_audio = self._generate_verb_form_audio(source_item, card, audio_failed_items)
            example_audios = tuple(
                self._generate_one_audio(
                    source_item=source_item,
                    text=example.example_sentence_nl,
                    field_name=f"Example_{index}_Audio",
                    label=f"example-{example.kind.value.replace('_', '-')}",
                    audio_failed_items=audio_failed_items,
                )
                for index, example in enumerate(card.ordered_form_examples(), start=1)
            )
            has_audio = (
                word_audio is not None
                or plural_audio is not None
                or (
                    verb_form_audio is not None
                    and any(verb_audio is not None for verb_audio in verb_form_audio.paths())
                )
                or any(example_audio is not None for example_audio in example_audios)
            )
            if not has_audio:
                continue

            note_audio = NoteAudio(
                word_audio=word_audio,
                plural_audio=plural_audio,
                verb_form_audio=verb_form_audio,
                example_audios=example_audios,
            )
            guid = build_note_guid(source_item)
            if source_item.concept is None:
                audio_by_guid[guid] = note_audio
                continue

            slots = grouped_audio.setdefault(
                guid,
                [None] * len(source_item.concept.dutch_answers),
            )
            slots[source_item.answer_index] = note_audio

        audio_by_guid.update(
            {guid: tuple(answer_audio) for guid, answer_audio in grouped_audio.items()}
        )
        return audio_by_guid

    def _generate_verb_form_audio(
        self,
        source_item: SourceItem,
        card: GeneratedCard,
        audio_failed_items: list[AudioFailedItem],
    ) -> VerbFormAudio | None:
        """Generate audio for editable verb-form fields."""
        if card.verb_forms is None:
            return None

        verb_forms = card.verb_forms
        return VerbFormAudio(
            infinitive_audio=self._generate_one_audio(
                source_item=source_item,
                text=verb_forms.infinitive,
                field_name="Verb_Infinitive_Audio",
                label="verb-infinitive",
                audio_failed_items=audio_failed_items,
            ),
            present_ik_audio=self._generate_one_audio(
                source_item=source_item,
                text=verb_forms.present_ik,
                field_name="Verb_Present_Ik_Audio",
                label="verb-present-ik",
                audio_failed_items=audio_failed_items,
            ),
            present_hij_audio=self._generate_one_audio(
                source_item=source_item,
                text=verb_forms.present_hij,
                field_name="Verb_Present_Hij_Audio",
                label="verb-present-hij",
                audio_failed_items=audio_failed_items,
            ),
            past_audio=self._generate_one_audio(
                source_item=source_item,
                text=verb_forms.past_tense,
                field_name="Verb_Past_Audio",
                label="verb-past",
                audio_failed_items=audio_failed_items,
            ),
            perfect_audio=self._generate_one_audio(
                source_item=source_item,
                text=verb_forms.perfect_tense,
                field_name="Verb_Perfect_Audio",
                label="verb-perfect",
                audio_failed_items=audio_failed_items,
            ),
        )

    def _generate_one_audio(
        self,
        *,
        source_item: SourceItem,
        text: str,
        field_name: str,
        label: str,
        audio_failed_items: list[AudioFailedItem],
    ) -> Path | None:
        """Generate one audio file and record non-fatal failures."""
        if self.audio_generator is None:
            return None

        try:
            return self.audio_generator.generate_audio(text, label=label)
        except (AudioGenerationError, OSError, ValueError) as exc:
            LOGGER.error("Failed to generate %s for '%s': %s", field_name, source_item.text, exc)
            audio_failed_items.append(
                AudioFailedItem(
                    source_word=source_item.text,
                    field_name=field_name,
                    reason=str(exc),
                )
            )
            return None

    def _generate_pending_items(
        self,
        *,
        pending_items: list[tuple[int, SourceItem]],
        total_items: int,
        cards_by_index: dict[int, tuple[SourceItem, GeneratedCard]],
        failed_items: list[FailedItem],
    ) -> None:
        """Generate all cache misses as one coordinated, resumable batch."""
        if not pending_items:
            return

        LOGGER.info(
            "Generating one coordinated batch for %s uncached item(s).",
            len(pending_items),
        )
        pending_by_index = dict(pending_items)
        existing_cards = [cards_by_index[index] for index in sorted(cards_by_index)]

        def accept_card(index: int, card: GeneratedCard) -> None:
            source_item = pending_by_index.get(index)
            if source_item is None:
                LOGGER.warning("Ignoring generated card for unknown source ID %s.", index)
                return
            self.cache.set(
                source_item,
                card,
                model_name=self.settings.llm.model_name,
                prompt_version=PROMPT_VERSION,
            )
            cards_by_index[index] = (source_item, card)
            LOGGER.info("Completed %s/%s: %s", index, total_items, source_item.text)

        try:
            result = self.llm_client.generate_cards(
                pending_items,
                existing_cards,
                on_card_accepted=accept_card,
            )
        except (LLMClientError, ValueError) as exc:
            LOGGER.error("Batch generation failed: %s", exc)
            result = BatchGenerationResult(
                failures={
                    index: str(exc)
                    for index in pending_by_index
                    if index not in cards_by_index
                },
            )

        for index, card in result.cards.items():
            if index not in cards_by_index:
                accept_card(index, card)

        for index, source_item in pending_items:
            if index in cards_by_index:
                continue
            reason = result.failures.get(index, "card remained unresolved after all attempts")
            LOGGER.error("Failed to process '%s': %s", source_item.text, reason)
            failed_items.append(FailedItem(source_word=source_item.text, reason=reason))


def load_source_items(
    input_path: Path,
    *,
    topic: str | None,
    lesson: str | None,
    exam_level: str | None,
) -> list[SourceItem]:
    """Load source concepts and flatten their Dutch answers for generation."""
    if not input_path.exists():
        raise FileNotFoundError(f"input file not found: {input_path}")

    source_items: list[SourceItem] = []
    identity_lines: dict[str, int] = {}
    concept_identity_lines: dict[str, int] = {}
    for line_number, raw_line in enumerate(input_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        source_concept = parse_source_concept_line(
            line,
            line_number=line_number,
            topic=topic,
            lesson=lesson,
            exam_level=exam_level,
        )

        concept_identity = source_concept.identity_key()
        if concept_identity in concept_identity_lines:
            raise ValueError(
                "duplicate source identity on lines "
                f"{concept_identity_lines[concept_identity]} and {line_number}; "
                "add distinct [id] prefixes"
            )
        concept_identity_lines[concept_identity] = line_number

        for source_item in source_concept.source_items():
            identity = source_item.identity_key()
            if identity in identity_lines:
                raise ValueError(
                    f"duplicate source identity on lines {identity_lines[identity]} and {line_number}; "
                    "add distinct [id] prefixes"
                )
            identity_lines[identity] = line_number
            source_items.append(source_item)

    if not source_items:
        raise ValueError("input file does not contain any vocabulary items")
    return source_items


def parse_source_item_line(
    line: str,
    *,
    line_number: int,
    topic: str | None,
    lesson: str | None,
    exam_level: str | None,
) -> SourceItem:
    """Parse one non-empty input line into a source item."""
    source_concept = parse_source_concept_line(
        line,
        line_number=line_number,
        topic=topic,
        lesson=lesson,
        exam_level=exam_level,
    )
    source_items = source_concept.source_items()
    if len(source_items) != 1:
        raise ValueError(
            f"grouped vocabulary item on line {line_number} expands to multiple source items; "
            "use load_source_items"
        )
    return source_items[0]


def parse_source_concept_line(
    line: str,
    *,
    line_number: int,
    topic: str | None,
    lesson: str | None,
    exam_level: str | None,
) -> SourceConcept:
    """Parse one authored line into a learner-facing vocabulary concept."""
    entry_id: str | None = None
    vocabulary_line = line
    if line.startswith("["):
        match = EXPLICIT_ID_PATTERN.fullmatch(line)
        if match is None:
            raise ValueError(
                f"invalid explicit ID on line {line_number}; expected '[id] <Dutch item>'"
            )
        entry_id = match.group(1).strip()
        vocabulary_line = match.group(2).strip()

    if (
        vocabulary_line == "-"
        or vocabulary_line.startswith("- ")
        or vocabulary_line.endswith(" -")
    ):
        raise ValueError(
            f"invalid hinted vocabulary item on line {line_number}; "
            "expected '<Dutch item> - <Russian translation hint>'"
        )

    translation_hint: str | None = None
    dutch_text = vocabulary_line
    if SOURCE_TRANSLATION_DELIMITER in vocabulary_line:
        dutch_text, translation_hint = vocabulary_line.split(SOURCE_TRANSLATION_DELIMITER, 1)
        dutch_text = dutch_text.strip()
        translation_hint = translation_hint.strip()
        if not dutch_text or not translation_hint:
            raise ValueError(
                f"invalid hinted vocabulary item on line {line_number}; "
                "expected '<Dutch item> - <Russian translation hint>'"
            )

    raw_answers = dutch_text.split("|")
    dutch_answers = tuple(answer.strip() for answer in raw_answers)
    if "|" in dutch_text and dutch_text != SOURCE_ALTERNATIVE_DELIMITER.join(dutch_answers):
        raise ValueError(
            f"invalid Dutch alternatives on line {line_number}; "
            "separate answers with the exact delimiter ' | '"
        )
    if any(not answer for answer in dutch_answers):
        raise ValueError(
            f"invalid Dutch alternatives on line {line_number}; "
            "each side of ' | ' must contain a Dutch answer"
        )

    normalized_answers = [" ".join(answer.casefold().split()) for answer in dutch_answers]
    if len(normalized_answers) != len(set(normalized_answers)):
        raise ValueError(f"duplicate Dutch alternative on line {line_number}")
    if len(dutch_answers) > 1 and translation_hint is None:
        raise ValueError(
            f"grouped vocabulary item on line {line_number} requires a Russian translation hint"
        )

    return SourceConcept(
        entry_id=entry_id,
        dutch_answers=dutch_answers,
        translation_hint=translation_hint,
        topic=topic,
        lesson=lesson,
        exam_level=exam_level,
    )


def resolve_deck_name(
    *,
    output_path: Path,
    settings: AppSettings,
    topic: str | None,
    lesson: str | None,
) -> str:
    """Resolve the final deck name."""
    if settings.deck.deck_name:
        return settings.deck.deck_name
    if lesson and topic:
        return f"{lesson} - {topic}"
    if lesson:
        return lesson
    if topic:
        return topic
    return output_path.stem
