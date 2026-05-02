"""End-to-end generation pipeline."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor, as_completed
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from .anki import NoteAudio, build_deck_package, build_note_guid
from .audio import AudioGenerationError, AudioGenerator, build_audio_generator
from .cache import CardCache
from .config import AppSettings
from .llm_client import LLMClient, LLMClientError
from .models import GeneratedCard, SourceItem
from .prompts import PROMPT_VERSION

LOGGER = logging.getLogger(__name__)


class CardGenerator(Protocol):
    """Protocol for card generation backends."""

    def generate_card(self, source_item: SourceItem) -> GeneratedCard:
        """Generate a validated card for one source item."""


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
    failed_items: list[FailedItem] = field(default_factory=list)
    audio_failed_items: list[AudioFailedItem] = field(default_factory=list)


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
        cards_by_index: dict[int, tuple[SourceItem, GeneratedCard]] = {}
        failed_items: list[FailedItem] = []
        cached_items = 0
        pending_items: list[tuple[int, SourceItem]] = []

        for index, source_item in enumerate(source_items, start=1):
            LOGGER.info("Preparing %s/%s: %s", index, len(source_items), source_item.text)
            try:
                card = None if force else self.cache.get(
                    source_item,
                    model_name=self.settings.llm.model_name,
                    prompt_version=PROMPT_VERSION,
                )
                if card is not None:
                    cached_items += 1
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
        if not cards:
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
            total_items=len(source_items),
            generated_items=len(cards),
            cached_items=cached_items,
            failed_items=failed_items,
            audio_failed_items=audio_failed_items,
        )

    def _generate_audio_for_cards(
        self,
        cards: list[tuple[SourceItem, GeneratedCard]],
        audio_failed_items: list[AudioFailedItem],
    ) -> dict[str, NoteAudio]:
        """Generate audio media for cards and return note GUID to audio mappings."""
        if not self.settings.audio.enabled or self.audio_generator is None:
            return {}

        audio_by_guid: dict[str, NoteAudio] = {}
        for source_item, card in cards:
            LOGGER.info("Generating audio for '%s'.", source_item.text)
            word_audio = self._generate_one_audio(
                source_item=source_item,
                text=card.dutch_word,
                field_name="Word_Audio",
                label="word",
                audio_failed_items=audio_failed_items,
            )
            example_audio = self._generate_one_audio(
                source_item=source_item,
                text=card.example_sentence_nl,
                field_name="Example_Audio",
                label="example",
                audio_failed_items=audio_failed_items,
            )
            if word_audio is not None or example_audio is not None:
                audio_by_guid[build_note_guid(source_item)] = NoteAudio(
                    word_audio=word_audio,
                    example_audio=example_audio,
                )
        return audio_by_guid

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
        """Generate uncached items, optionally in parallel, while preserving input order."""
        if not pending_items:
            return

        worker_count = min(self.settings.generation.parallelism, len(pending_items))
        LOGGER.info(
            "Generating %s uncached items with up to %s parallel worker(s).",
            len(pending_items),
            worker_count,
        )

        if worker_count == 1:
            for index, source_item in pending_items:
                self._process_pending_item(
                    index=index,
                    source_item=source_item,
                    total_items=total_items,
                    cards_by_index=cards_by_index,
                    failed_items=failed_items,
                )
            return

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures: dict[Future[GeneratedCard], tuple[int, SourceItem]] = {
                executor.submit(self._generate_and_cache_card, source_item): (index, source_item)
                for index, source_item in pending_items
            }
            for future in as_completed(futures):
                index, source_item = futures[future]
                try:
                    card = future.result()
                    cards_by_index[index] = (source_item, card)
                    LOGGER.info("Completed %s/%s: %s", index, total_items, source_item.text)
                except (LLMClientError, ValueError) as exc:
                    LOGGER.error("Failed to process '%s': %s", source_item.text, exc)
                    failed_items.append(FailedItem(source_word=source_item.text, reason=str(exc)))

    def _process_pending_item(
        self,
        *,
        index: int,
        source_item: SourceItem,
        total_items: int,
        cards_by_index: dict[int, tuple[SourceItem, GeneratedCard]],
        failed_items: list[FailedItem],
    ) -> None:
        """Generate and cache one uncached item in sequential mode."""
        try:
            card = self._generate_and_cache_card(source_item)
            cards_by_index[index] = (source_item, card)
            LOGGER.info("Completed %s/%s: %s", index, total_items, source_item.text)
        except (LLMClientError, ValueError) as exc:
            LOGGER.error("Failed to process '%s': %s", source_item.text, exc)
            failed_items.append(FailedItem(source_word=source_item.text, reason=str(exc)))

    def _generate_and_cache_card(self, source_item: SourceItem) -> GeneratedCard:
        """Generate one card and persist it in the cache."""
        card = self.llm_client.generate_card(source_item)
        self.cache.set(
            source_item,
            card,
            model_name=self.settings.llm.model_name,
            prompt_version=PROMPT_VERSION,
        )
        return card


def load_source_items(
    input_path: Path,
    *,
    topic: str | None,
    lesson: str | None,
    exam_level: str | None,
) -> list[SourceItem]:
    """Load source items from a plain text file, one item per line."""
    if not input_path.exists():
        raise FileNotFoundError(f"input file not found: {input_path}")

    source_items: list[SourceItem] = []
    for raw_line in input_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        source_items.append(SourceItem(text=line, topic=topic, lesson=lesson, exam_level=exam_level))

    if not source_items:
        raise ValueError("input file does not contain any vocabulary items")
    return source_items


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
