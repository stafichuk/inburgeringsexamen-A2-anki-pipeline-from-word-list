from collections.abc import Callable
from pathlib import Path

import pytest

import app.pipeline as pipeline_module
from app.audio import AudioGenerationError
from app.cache import CardCache
from app.config import AppSettings
from app.llm_client import BatchGenerationResult
from app.models import GeneratedCard, SourceItem, VerbForms
from app.pipeline import DeckGenerationPipeline, load_source_items


def make_settings(cache_dir: Path, *, audio_enabled: bool = False) -> AppSettings:
    payload = {
        "llm": {
            "base_url": "https://example.invalid/v1/chat/completions",
            "api_token": "token",
            "model_name": "test-model",
        },
        "cache": {"directory": str(cache_dir)},
    }
    if audio_enabled:
        payload["audio"] = {
            "enabled": True,
            "directory": str(cache_dir.parent / ".audio"),
            "azure": {
                "region": "westeurope",
                "api_key": "key",
                "voice": "nl-NL-FennaNeural",
            },
        }
    return AppSettings.model_validate(payload)


def make_card(word: str) -> GeneratedCard:
    present_ik = "ik leer" if word == "leren" else f"ik {word}"
    present_hij = "hij leert" if word == "leren" else f"hij {word}t"
    perfect_tense = "heeft geleerd" if word == "leren" else f"heeft {word} gebruikt"
    return GeneratedCard(
        dutch_word=word,
        russian_translation="учиться",
        part_of_speech="verb",
        ipa_transcription="ˈleːrə(n)",
        lesson_topic="De school",
        form_examples=[
            {
                "kind": "present_tense",
                "form": present_ik,
                "example_sentence_nl": f"{present_ik.capitalize()} in de les.",
                "example_sentence_ru": f"Я использую {word} на уроке.",
            },
            {
                "kind": "past_tense",
                "form": "gebruikte",
                "example_sentence_nl": f"Ik gebruikte {word} gisteren.",
                "example_sentence_ru": f"Вчера я использовал {word}.",
            },
            {
                "kind": "perfect_tense",
                "form": perfect_tense,
                "example_sentence_nl": f"Hij {perfect_tense}.",
                "example_sentence_ru": f"Я использовал {word}.",
            },
        ],
        tags=["school", "verb"],
        verb_forms=VerbForms(
            infinitive=word,
            present_ik=present_ik,
            present_hij=present_hij,
            past_tense="leerde",
            perfect_tense=perfect_tense,
        ),
    )


def make_noun_card(word: str) -> GeneratedCard:
    return GeneratedCard(
        dutch_word=f"de {word}",
        russian_translation="тётя",
        part_of_speech="noun",
        ipa_transcription="ˈtɑn.tə",
        lesson_topic="De familie",
        form_examples=[
            {
                "kind": "singular",
                "form": word,
                "example_sentence_nl": f"Mijn {word} woont in Amsterdam.",
                "example_sentence_ru": "Моя тётя живёт в Амстердаме.",
            },
            {
                "kind": "plural",
                "form": f"{word}s",
                "example_sentence_nl": f"Mijn twee {word}s komen op bezoek.",
                "example_sentence_ru": "Мои две тёти придут в гости.",
            },
        ],
        tags=["familie", "noun"],
        plural_form=f"{word}s",
        front_hint="тётя (множественное число?)",
    )


def make_alternative_card(source_item: SourceItem) -> GeneratedCard:
    """Build a schema-valid leaf card without coupling grouping tests to noun rules."""
    translation = source_item.translation_hint or "перевод"
    return GeneratedCard(
        dutch_word=source_item.text,
        russian_translation=translation,
        part_of_speech="phrase",
        ipa_transcription="test-ipa",
        lesson_topic=source_item.topic or "A2",
        form_examples=[
            {
                "kind": "default",
                "form": source_item.text,
                "example_sentence_nl": f"Ik ken {source_item.text}.",
                "example_sentence_ru": f"Я знаю выражение: {translation}.",
            }
        ],
        tags=["alternative"],
    )


class FakeClient:
    def __init__(self) -> None:
        self.calls = 0
        self.pending_batches: list[list[str]] = []
        self.existing_contexts: list[list[str]] = []

    def make_card_for(self, source_item: SourceItem) -> GeneratedCard:
        return make_card(source_item.text)

    def generate_cards(
        self,
        pending_items: list[tuple[int, SourceItem]],
        existing_cards: list[tuple[SourceItem, GeneratedCard]],
        on_card_accepted: Callable[[int, GeneratedCard], None] | None = None,
    ) -> BatchGenerationResult:
        self.calls += 1
        self.pending_batches.append([source_item.text for _, source_item in pending_items])
        self.existing_contexts.append([source_item.text for source_item, _ in existing_cards])
        cards = {
            source_id: self.make_card_for(source_item)
            for source_id, source_item in pending_items
        }
        if on_card_accepted is not None:
            for source_id, card in cards.items():
                on_card_accepted(source_id, card)
        return BatchGenerationResult(cards=cards)


class NounClient(FakeClient):
    def make_card_for(self, source_item: SourceItem) -> GeneratedCard:
        return make_noun_card(source_item.text)


class TranslationHintNounClient(FakeClient):
    def make_card_for(self, source_item: SourceItem) -> GeneratedCard:
        translation = source_item.translation_hint or "родственник"
        bare_form = source_item.text.removeprefix("de ").removeprefix("het ")
        return GeneratedCard(
            dutch_word=source_item.text,
            russian_translation=translation,
            part_of_speech="noun",
            ipa_transcription="neːf",
            lesson_topic="Familie",
            form_examples=[
                {
                    "kind": "singular",
                    "form": bare_form,
                    "example_sentence_nl": f"Mijn {bare_form} komt vandaag.",
                    "example_sentence_ru": f"{translation.capitalize()} придет сегодня.",
                },
                {
                    "kind": "plural",
                    "form": "neven",
                    "example_sentence_nl": "Mijn neven komen vandaag.",
                    "example_sentence_ru": "Мои родственники придут сегодня.",
                },
            ],
            tags=["familie", "noun"],
            plural_form="neven",
            front_hint=f"{translation} (множественное число?)",
        )


class AlternativeClient(FakeClient):
    def make_card_for(self, source_item: SourceItem) -> GeneratedCard:
        return make_alternative_card(source_item)


class PartiallyFailingAlternativeClient(AlternativeClient):
    def __init__(self, failing_word: str) -> None:
        super().__init__()
        self.failing_word = failing_word

    def generate_cards(
        self,
        pending_items: list[tuple[int, SourceItem]],
        existing_cards: list[tuple[SourceItem, GeneratedCard]],
        on_card_accepted: Callable[[int, GeneratedCard], None] | None = None,
    ) -> BatchGenerationResult:
        self.calls += 1
        self.pending_batches.append([source_item.text for _, source_item in pending_items])
        self.existing_contexts.append([source_item.text for source_item, _ in existing_cards])
        cards = {
            source_id: self.make_card_for(source_item)
            for source_id, source_item in pending_items
            if source_item.text != self.failing_word
        }
        failures = {
            source_id: "invalid alternative payload"
            for source_id, source_item in pending_items
            if source_item.text == self.failing_word
        }
        if on_card_accepted is not None:
            for source_id, card in cards.items():
                on_card_accepted(source_id, card)
        return BatchGenerationResult(cards=cards, failures=failures)


class PartiallyFailingClient(FakeClient):
    def generate_cards(
        self,
        pending_items: list[tuple[int, SourceItem]],
        existing_cards: list[tuple[SourceItem, GeneratedCard]],
        on_card_accepted: Callable[[int, GeneratedCard], None] | None = None,
    ) -> BatchGenerationResult:
        self.calls += 1
        self.pending_batches.append([source_item.text for _, source_item in pending_items])
        self.existing_contexts.append([source_item.text for source_item, _ in existing_cards])
        cards = {
            source_id: make_card(source_item.text)
            for source_id, source_item in pending_items
            if source_item.text != "fout"
        }
        failures = {
            source_id: "invalid payload"
            for source_id, source_item in pending_items
            if source_item.text == "fout"
        }
        if on_card_accepted is not None:
            for source_id, card in cards.items():
                on_card_accepted(source_id, card)
        return BatchGenerationResult(cards=cards, failures=failures)


class ReverseOrderClient(FakeClient):
    def generate_cards(
        self,
        pending_items: list[tuple[int, SourceItem]],
        existing_cards: list[tuple[SourceItem, GeneratedCard]],
        on_card_accepted: Callable[[int, GeneratedCard], None] | None = None,
    ) -> BatchGenerationResult:
        self.calls += 1
        reversed_items = list(reversed(pending_items))
        cards = {
            source_id: make_card(source_item.text)
            for source_id, source_item in reversed_items
        }
        if on_card_accepted is not None:
            for source_id, card in cards.items():
                on_card_accepted(source_id, card)
        return BatchGenerationResult(cards=cards)


class InterruptingAfterAcceptanceClient(FakeClient):
    def generate_cards(
        self,
        pending_items: list[tuple[int, SourceItem]],
        existing_cards: list[tuple[SourceItem, GeneratedCard]],
        on_card_accepted: Callable[[int, GeneratedCard], None] | None = None,
    ) -> BatchGenerationResult:
        self.calls += 1
        self.pending_batches.append([source_item.text for _, source_item in pending_items])
        self.existing_contexts.append([source_item.text for source_item, _ in existing_cards])
        source_id, source_item = pending_items[0]
        card = make_card(source_item.text)
        if on_card_accepted is not None:
            on_card_accepted(source_id, card)
        raise pipeline_module.LLMClientError("connection failed during retry")


class FakeAudioGenerator:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.calls: list[tuple[str, str]] = []

    def generate_audio(self, text: str, *, label: str) -> Path:
        self.calls.append((label, text))
        path = self.directory / f"{label}-{len(self.calls)}.mp3"
        path.write_bytes(f"{label}:{text}".encode("utf-8"))
        return path


class FailingExampleAudioGenerator(FakeAudioGenerator):
    def generate_audio(self, text: str, *, label: str) -> Path:
        if label == "example-past-tense":
            raise AudioGenerationError("example synthesis failed")
        return super().generate_audio(text, label=label)


class FailingVerbFormAudioGenerator(FakeAudioGenerator):
    def generate_audio(self, text: str, *, label: str) -> Path:
        if label == "verb-perfect":
            raise AudioGenerationError("verb form synthesis failed")
        return super().generate_audio(text, label=label)


class FailingOneAlternativeAudioGenerator(FakeAudioGenerator):
    def generate_audio(self, text: str, *, label: str) -> Path:
        if "de kleren" in text:
            raise AudioGenerationError("alternative synthesis failed")
        return super().generate_audio(text, label=label)


def test_pipeline_uses_cache_without_calling_llm(tmp_path: Path) -> None:
    input_path = tmp_path / "words.txt"
    input_path.write_text("leren\n", encoding="utf-8")
    output_path = tmp_path / "deck.apkg"
    settings = make_settings(tmp_path / ".cache")

    first_client = FakeClient()
    pipeline = DeckGenerationPipeline(settings, llm_client=first_client)
    first_result = pipeline.run(input_path=input_path, output_path=output_path)
    assert first_result.generated_items == 1
    assert first_client.calls == 1
    assert first_client.pending_batches == [["leren"]]
    assert first_client.existing_contexts == [[]]

    second_client = FakeClient()
    second_pipeline = DeckGenerationPipeline(settings, llm_client=second_client, cache=CardCache(settings.cache.directory))
    second_result = second_pipeline.run(input_path=input_path, output_path=tmp_path / "deck-2.apkg")

    assert second_result.cached_items == 1
    assert second_client.calls == 0


def test_pipeline_generates_only_added_words_with_cached_cards_as_context(tmp_path: Path) -> None:
    input_path = tmp_path / "words.txt"
    input_path.write_text("leren\nwerken\n", encoding="utf-8")
    settings = make_settings(tmp_path / ".cache")

    DeckGenerationPipeline(settings, llm_client=FakeClient()).run(
        input_path=input_path,
        output_path=tmp_path / "first.apkg",
    )

    input_path.write_text("leren\nwerken\nwonen\n", encoding="utf-8")
    incremental_client = FakeClient()
    result = DeckGenerationPipeline(settings, llm_client=incremental_client).run(
        input_path=input_path,
        output_path=tmp_path / "second.apkg",
    )

    assert result.deck_written is True
    assert result.cached_items == 2
    assert incremental_client.pending_batches == [["wonen"]]
    assert incremental_client.existing_contexts == [["leren", "werken"]]


def test_pipeline_force_bypasses_cache(tmp_path: Path) -> None:
    input_path = tmp_path / "words.txt"
    input_path.write_text("leren\n", encoding="utf-8")
    output_path = tmp_path / "deck.apkg"
    settings = make_settings(tmp_path / ".cache")

    first_client = FakeClient()
    DeckGenerationPipeline(settings, llm_client=first_client).run(input_path=input_path, output_path=output_path)

    forced_client = FakeClient()
    result = DeckGenerationPipeline(settings, llm_client=forced_client).run(
        input_path=input_path,
        output_path=tmp_path / "forced.apkg",
        force=True,
    )

    assert result.cached_items == 0
    assert forced_client.calls == 1
    assert forced_client.pending_batches == [["leren"]]
    assert forced_client.existing_contexts == [[]]


def test_pipeline_caches_partial_success_but_preserves_existing_deck(tmp_path: Path) -> None:
    input_path = tmp_path / "words.txt"
    input_path.write_text("leren\nfout\n", encoding="utf-8")
    output_path = tmp_path / "deck.apkg"
    output_path.write_text("previous complete deck", encoding="utf-8")
    settings = make_settings(tmp_path / ".cache")

    client = PartiallyFailingClient()
    result = DeckGenerationPipeline(settings, llm_client=client).run(input_path=input_path, output_path=output_path)

    assert output_path.read_text(encoding="utf-8") == "previous complete deck"
    assert result.deck_written is False
    assert result.generated_items == 1
    assert len(result.failed_items) == 1
    assert result.failed_items[0].source_word == "fout"

    retry_client = FakeClient()
    retry_result = DeckGenerationPipeline(settings, llm_client=retry_client).run(
        input_path=input_path,
        output_path=output_path,
    )

    assert retry_result.deck_written is True
    assert retry_client.pending_batches == [["fout"]]
    assert retry_client.existing_contexts == [["leren"]]


def test_pipeline_persists_an_accepted_card_before_a_later_batch_error(tmp_path: Path) -> None:
    input_path = tmp_path / "words.txt"
    input_path.write_text("leren\nwerken\n", encoding="utf-8")
    output_path = tmp_path / "deck.apkg"
    output_path.write_text("previous complete deck", encoding="utf-8")
    settings = make_settings(tmp_path / ".cache")

    result = DeckGenerationPipeline(
        settings,
        llm_client=InterruptingAfterAcceptanceClient(),
    ).run(
        input_path=input_path,
        output_path=output_path,
    )

    assert result.deck_written is False
    assert result.generated_items == 1
    assert output_path.read_text(encoding="utf-8") == "previous complete deck"

    retry_client = FakeClient()
    retry_result = DeckGenerationPipeline(settings, llm_client=retry_client).run(
        input_path=input_path,
        output_path=output_path,
    )

    assert retry_result.deck_written is True
    assert retry_client.pending_batches == [["werken"]]
    assert retry_client.existing_contexts == [["leren"]]


def test_force_partial_refresh_discards_failed_old_cache_and_resumes(tmp_path: Path) -> None:
    input_path = tmp_path / "words.txt"
    input_path.write_text("leren\nfout\n", encoding="utf-8")
    output_path = tmp_path / "deck.apkg"
    settings = make_settings(tmp_path / ".cache")

    DeckGenerationPipeline(settings, llm_client=FakeClient()).run(
        input_path=input_path,
        output_path=output_path,
    )
    previous_deck = output_path.read_bytes()

    forced_result = DeckGenerationPipeline(
        settings,
        llm_client=PartiallyFailingClient(),
    ).run(
        input_path=input_path,
        output_path=output_path,
        force=True,
    )

    assert forced_result.deck_written is False
    assert forced_result.cached_items == 0
    assert output_path.read_bytes() == previous_deck

    retry_client = FakeClient()
    retry_result = DeckGenerationPipeline(settings, llm_client=retry_client).run(
        input_path=input_path,
        output_path=output_path,
    )

    assert retry_result.deck_written is True
    assert retry_client.pending_batches == [["fout"]]
    assert retry_client.existing_contexts == [["leren"]]


def test_force_refresh_marker_suppresses_all_old_cards_after_interruption(tmp_path: Path) -> None:
    input_path = tmp_path / "words.txt"
    input_path.write_text("leren\nwerken\n", encoding="utf-8")
    settings = make_settings(tmp_path / ".cache")

    DeckGenerationPipeline(settings, llm_client=FakeClient()).run(
        input_path=input_path,
        output_path=tmp_path / "first.apkg",
    )

    cache = CardCache(settings.cache.directory)
    source_items = load_source_items(
        input_path,
        topic=settings.generation.default_topic,
        lesson=settings.generation.default_lesson,
        exam_level=settings.generation.default_exam_level,
    )
    cache.begin_refresh(source_items)

    resumed_client = FakeClient()
    result = DeckGenerationPipeline(
        settings,
        llm_client=resumed_client,
        cache=cache,
    ).run(
        input_path=input_path,
        output_path=tmp_path / "resumed.apkg",
    )

    assert result.deck_written is True
    assert result.cached_items == 0
    assert resumed_client.pending_batches == [["leren", "werken"]]
    assert resumed_client.existing_contexts == [[]]


def test_pipeline_preserves_input_order_when_batch_response_is_reversed(tmp_path: Path, monkeypatch) -> None:
    input_path = tmp_path / "words.txt"
    input_path.write_text("eerste\ntweede\nderde\n", encoding="utf-8")
    output_path = tmp_path / "deck.apkg"
    settings = make_settings(tmp_path / ".cache")
    captured_order: list[str] = []

    def fake_build_deck_package(cards, output_path, deck_name, settings, audio_by_guid=None):  # type: ignore[no-untyped-def]
        captured_order.extend(source_item.text for source_item, _ in cards)
        output_path.write_text("stub", encoding="utf-8")
        return output_path

    monkeypatch.setattr(pipeline_module, "build_deck_package", fake_build_deck_package)

    client = ReverseOrderClient()
    result = DeckGenerationPipeline(settings, llm_client=client).run(input_path=input_path, output_path=output_path)

    assert result.generated_items == 3
    assert captured_order == ["eerste", "tweede", "derde"]


def test_pipeline_keeps_duplicate_words_with_distinct_translation_hints(tmp_path: Path, monkeypatch) -> None:
    input_path = tmp_path / "words.txt"
    input_path.write_text("de neef - племянник\nde neef - двоюродный брат\n", encoding="utf-8")
    output_path = tmp_path / "deck.apkg"
    settings = make_settings(tmp_path / ".cache")
    captured_cards: list[tuple[str, str | None, str]] = []

    def fake_build_deck_package(cards, output_path, deck_name, settings, audio_by_guid=None):  # type: ignore[no-untyped-def]
        captured_cards.extend(
            (source_item.text, source_item.translation_hint, card.russian_translation)
            for source_item, card in cards
        )
        output_path.write_text("stub", encoding="utf-8")
        return output_path

    monkeypatch.setattr(pipeline_module, "build_deck_package", fake_build_deck_package)

    client = TranslationHintNounClient()
    result = DeckGenerationPipeline(settings, llm_client=client).run(
        input_path=input_path,
        output_path=output_path,
    )

    assert result.generated_items == 2
    assert result.cached_items == 0
    assert client.calls == 1
    assert captured_cards == [
        ("de neef", "племянник", "племянник"),
        ("de neef", "двоюродный брат", "двоюродный брат"),
    ]


def test_load_source_items_parses_plain_and_hinted_lines(tmp_path: Path) -> None:
    input_path = tmp_path / "words.txt"
    input_path.write_text(
        "\n".join(
            [
                "# Familie",
                "",
                "de broer",
                "[neef-nephew] de neef - племянник",
                "[neef-cousin] de neef - двоюродный брат",
                "e-mail",
                "kinderopvang - детский сад",
            ]
        ),
        encoding="utf-8",
    )

    items = load_source_items(input_path, topic="Familie", lesson="Les 1", exam_level="A2")

    assert [(item.text, item.translation_hint) for item in items] == [
        ("de broer", None),
        ("de neef", "племянник"),
        ("de neef", "двоюродный брат"),
        ("e-mail", None),
        ("kinderopvang", "детский сад"),
    ]
    assert [item.entry_id for item in items] == [
        None,
        "neef-nephew",
        "neef-cousin",
        None,
        None,
    ]
    assert all(item.topic == "Familie" for item in items)


def test_load_source_items_rejects_duplicate_identities(tmp_path: Path) -> None:
    input_path = tmp_path / "words.txt"
    input_path.write_text("de broer\nDe   broer\n", encoding="utf-8")

    try:
        load_source_items(input_path, topic="Familie", lesson="Les 1", exam_level="A2")
    except ValueError as exc:
        assert "duplicate source identity on lines 1 and 2" in str(exc)
        assert "[id]" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("duplicate identity should have failed")


def test_explicit_id_remains_stable_when_word_is_corrected() -> None:
    original = SourceItem(entry_id="friend", text="de vrient", topic="Familie", lesson="Les 1")
    corrected = SourceItem(entry_id="friend", text="de vriend", topic="Familie", lesson="Les 1")

    assert original.identity_key() == corrected.identity_key()


def test_load_source_items_rejects_malformed_translation_hints(tmp_path: Path) -> None:
    for bad_line in ("de neef -", "- племянник", " - племянник", "-"):
        input_path = tmp_path / f"{bad_line.replace(' ', '_')}.txt"
        input_path.write_text(f"{bad_line}\n", encoding="utf-8")

        try:
            load_source_items(input_path, topic=None, lesson=None, exam_level=None)
        except ValueError as exc:
            assert "expected '<Dutch item> - <Russian translation hint>'" in str(exc)
        else:  # pragma: no cover
            raise AssertionError(f"malformed line should have failed: {bad_line}")


def test_pipeline_generates_audio_for_successful_cards(tmp_path: Path, monkeypatch) -> None:
    input_path = tmp_path / "words.txt"
    input_path.write_text("leren\n", encoding="utf-8")
    output_path = tmp_path / "deck.apkg"
    settings = make_settings(tmp_path / ".cache", audio_enabled=True)
    captured_audio_by_guid = {}

    def fake_build_deck_package(cards, output_path, deck_name, settings, audio_by_guid=None):  # type: ignore[no-untyped-def]
        captured_audio_by_guid.update(audio_by_guid or {})
        output_path.write_text("stub", encoding="utf-8")
        return output_path

    monkeypatch.setattr(pipeline_module, "build_deck_package", fake_build_deck_package)

    audio_generator = FakeAudioGenerator(tmp_path / "audio")
    result = DeckGenerationPipeline(
        settings,
        llm_client=FakeClient(),
        audio_generator=audio_generator,
    ).run(input_path=input_path, output_path=output_path)

    assert result.generated_items == 1
    assert result.audio_failed_items == []
    assert audio_generator.calls == [
        ("word", "leren"),
        ("verb-infinitive", "leren"),
        ("verb-present-ik", "ik leer"),
        ("verb-present-hij", "hij leert"),
        ("verb-past", "leerde"),
        ("verb-perfect", "heeft geleerd"),
        ("example-present-tense", "Ik leer in de les."),
        ("example-past-tense", "Ik gebruikte leren gisteren."),
        ("example-perfect-tense", "Hij heeft geleerd."),
    ]
    assert len(captured_audio_by_guid) == 1
    audio = next(iter(captured_audio_by_guid.values()))
    assert audio.word_audio is not None
    assert audio.verb_form_audio is not None
    assert all(verb_audio is not None for verb_audio in audio.verb_form_audio.paths())
    assert len(audio.example_audios) == 3
    assert all(example_audio is not None for example_audio in audio.example_audios)


def test_pipeline_generates_noun_word_audio_with_article(tmp_path: Path, monkeypatch) -> None:
    input_path = tmp_path / "words.txt"
    input_path.write_text("tante\n", encoding="utf-8")
    output_path = tmp_path / "deck.apkg"
    settings = make_settings(tmp_path / ".cache", audio_enabled=True)
    captured_audio_by_guid = {}

    def fake_build_deck_package(cards, output_path, deck_name, settings, audio_by_guid=None):  # type: ignore[no-untyped-def]
        captured_audio_by_guid.update(audio_by_guid or {})
        output_path.write_text("stub", encoding="utf-8")
        return output_path

    monkeypatch.setattr(pipeline_module, "build_deck_package", fake_build_deck_package)

    audio_generator = FakeAudioGenerator(tmp_path / "audio")
    result = DeckGenerationPipeline(
        settings,
        llm_client=NounClient(),
        audio_generator=audio_generator,
    ).run(input_path=input_path, output_path=output_path)

    assert result.generated_items == 1
    assert audio_generator.calls[0] == ("word", "de tante")
    assert audio_generator.calls[1:] == [
        ("plural", "tantes"),
        ("example-singular", "Mijn tante woont in Amsterdam."),
        ("example-plural", "Mijn twee tantes komen op bezoek."),
    ]
    audio = next(iter(captured_audio_by_guid.values()))
    assert audio.plural_audio is not None


def test_pipeline_reports_partial_verb_form_audio_failures(tmp_path: Path, monkeypatch) -> None:
    input_path = tmp_path / "words.txt"
    input_path.write_text("leren\n", encoding="utf-8")
    output_path = tmp_path / "deck.apkg"
    settings = make_settings(tmp_path / ".cache", audio_enabled=True)
    captured_audio_by_guid = {}

    def fake_build_deck_package(cards, output_path, deck_name, settings, audio_by_guid=None):  # type: ignore[no-untyped-def]
        captured_audio_by_guid.update(audio_by_guid or {})
        output_path.write_text("stub", encoding="utf-8")
        return output_path

    monkeypatch.setattr(pipeline_module, "build_deck_package", fake_build_deck_package)

    result = DeckGenerationPipeline(
        settings,
        llm_client=FakeClient(),
        audio_generator=FailingVerbFormAudioGenerator(tmp_path / "audio"),
    ).run(input_path=input_path, output_path=output_path)

    assert output_path.exists()
    assert result.generated_items == 1
    assert len(result.audio_failed_items) == 1
    assert result.audio_failed_items[0].source_word == "leren"
    assert result.audio_failed_items[0].field_name == "Verb_Perfect_Audio"
    audio = next(iter(captured_audio_by_guid.values()))
    assert audio.verb_form_audio is not None
    assert audio.verb_form_audio.infinitive_audio is not None
    assert audio.verb_form_audio.present_ik_audio is not None
    assert audio.verb_form_audio.present_hij_audio is not None
    assert audio.verb_form_audio.past_audio is not None
    assert audio.verb_form_audio.perfect_audio is None


def test_pipeline_reports_partial_audio_failures(tmp_path: Path, monkeypatch) -> None:
    input_path = tmp_path / "words.txt"
    input_path.write_text("leren\n", encoding="utf-8")
    output_path = tmp_path / "deck.apkg"
    settings = make_settings(tmp_path / ".cache", audio_enabled=True)
    captured_audio_by_guid = {}

    def fake_build_deck_package(cards, output_path, deck_name, settings, audio_by_guid=None):  # type: ignore[no-untyped-def]
        captured_audio_by_guid.update(audio_by_guid or {})
        output_path.write_text("stub", encoding="utf-8")
        return output_path

    monkeypatch.setattr(pipeline_module, "build_deck_package", fake_build_deck_package)

    result = DeckGenerationPipeline(
        settings,
        llm_client=FakeClient(),
        audio_generator=FailingExampleAudioGenerator(tmp_path / "audio"),
    ).run(input_path=input_path, output_path=output_path)

    assert output_path.exists()
    assert result.generated_items == 1
    assert len(result.audio_failed_items) == 1
    assert result.audio_failed_items[0].source_word == "leren"
    assert result.audio_failed_items[0].field_name == "Example_2_Audio"
    audio = next(iter(captured_audio_by_guid.values()))
    assert audio.word_audio is not None
    assert audio.verb_form_audio is not None
    assert all(verb_audio is not None for verb_audio in audio.verb_form_audio.paths())
    assert audio.example_audios[0] is not None
    assert audio.example_audios[1] is None
    assert audio.example_audios[2] is not None


def test_load_source_items_expands_grouped_answers_with_shared_concept(tmp_path: Path) -> None:
    input_path = tmp_path / "words.txt"
    input_path.write_text(
        "[clothes] de kleding | de kleren - одежда\n",
        encoding="utf-8",
    )

    items = load_source_items(
        input_path,
        topic="Kleding",
        lesson="Les 1",
        exam_level="A2",
    )

    assert [item.text for item in items] == ["de kleding", "de kleren"]
    assert [item.answer_index for item in items] == [0, 1]
    assert [item.entry_id for item in items] == ["clothes", None]
    assert all(item.translation_hint == "одежда" for item in items)
    assert all(item.topic == "Kleding" for item in items)
    assert all(item.lesson == "Les 1" for item in items)
    assert all(item.exam_level == "A2" for item in items)

    concept = items[0].concept
    assert concept is not None
    assert items[1].concept is concept
    assert concept.entry_id == "clothes"
    assert concept.dutch_answers == ("de kleding", "de kleren")
    assert concept.translation_hint == "одежда"
    assert concept.source_text() == "de kleding | de kleren"
    assert [item.text for item in concept.source_items()] == ["de kleding", "de kleren"]
    assert items[0].accepted_dutch_answers == ("de kleding", "de kleren")
    assert items[1].accepted_dutch_answers == ("de kleding", "de kleren")
    assert items[0].concept_identity_key() == items[1].concept_identity_key()


@pytest.mark.parametrize(
    ("bad_line", "expected_fragment"),
    [
        ("de kleding | - одежда", "alternative"),
        ("de kleding |  | de kleren - одежда", "alternative"),
        ("de kleding|de kleren - одежда", " | "),
        ("de kleding  |  de kleren - одежда", " | "),
        ("de kleding | DE   KLEDING - одежда", "duplicate"),
        ("de kleding | de kleren", "translation hint"),
    ],
)
def test_load_source_items_rejects_invalid_grouped_answers(
    tmp_path: Path,
    bad_line: str,
    expected_fragment: str,
) -> None:
    input_path = tmp_path / "words.txt"
    input_path.write_text(f"{bad_line}\n", encoding="utf-8")

    with pytest.raises(ValueError) as exc_info:
        load_source_items(input_path, topic=None, lesson=None, exam_level=None)

    message = str(exc_info.value)
    assert "line 1" in message
    assert expected_fragment in message


def test_pipeline_adding_group_sibling_reuses_first_answer_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "words.txt"
    input_path.write_text("de kleding - одежда\n", encoding="utf-8")
    output_path = tmp_path / "deck.apkg"
    settings = make_settings(tmp_path / ".cache")
    published_answers: list[list[str]] = []

    def fake_build_deck_package(cards, output_path, deck_name, settings, audio_by_guid=None):  # type: ignore[no-untyped-def]
        published_answers.append([source_item.text for source_item, _ in cards])
        output_path.write_text("complete deck", encoding="utf-8")
        return output_path

    monkeypatch.setattr(pipeline_module, "build_deck_package", fake_build_deck_package)

    first_client = AlternativeClient()
    first_result = DeckGenerationPipeline(settings, llm_client=first_client).run(
        input_path=input_path,
        output_path=output_path,
    )
    assert first_result.total_items == 1
    assert first_result.generated_items == 1
    assert first_result.cached_items == 0

    input_path.write_text(
        "de kleding | de kleren - одежда\n",
        encoding="utf-8",
    )
    sibling_client = AlternativeClient()
    sibling_result = DeckGenerationPipeline(settings, llm_client=sibling_client).run(
        input_path=input_path,
        output_path=output_path,
    )

    assert sibling_result.total_items == 1
    assert sibling_result.generated_items == 1
    assert sibling_result.cached_items == 0
    assert sibling_client.pending_batches == [["de kleren"]]
    assert sibling_client.existing_contexts == [["de kleding"]]
    assert published_answers[-1] == ["de kleding", "de kleren"]

    cached_client = AlternativeClient()
    cached_result = DeckGenerationPipeline(settings, llm_client=cached_client).run(
        input_path=input_path,
        output_path=output_path,
    )

    assert cached_result.total_items == 1
    assert cached_result.generated_items == 1
    assert cached_result.cached_items == 1
    assert cached_client.calls == 0


def test_pipeline_regenerates_stale_cache_that_replaced_group_answer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "words.txt"
    input_path.write_text(
        "de kleding | de kleren - одежда\n",
        encoding="utf-8",
    )
    settings = make_settings(tmp_path / ".cache")
    cache = CardCache(settings.cache.directory)
    standalone = SourceItem(text="de kleding", translation_hint="одежда")
    stale_card = make_alternative_card(standalone).model_copy(
        update={"dutch_word": "de garderobe"}
    )
    cache.set(
        standalone,
        stale_card,
        model_name="old-model",
        prompt_version="old-prompt",
    )
    published_answers: list[str] = []

    def fake_build_deck_package(cards, output_path, deck_name, settings, audio_by_guid=None):  # type: ignore[no-untyped-def]
        published_answers.extend(card.dutch_word for _, card in cards)
        output_path.write_text("complete deck", encoding="utf-8")
        return output_path

    monkeypatch.setattr(pipeline_module, "build_deck_package", fake_build_deck_package)
    client = AlternativeClient()

    result = DeckGenerationPipeline(
        settings,
        llm_client=client,
        cache=cache,
    ).run(input_path=input_path, output_path=tmp_path / "deck.apkg")

    assert result.deck_written is True
    assert result.cached_items == 0
    assert client.pending_batches == [["de kleding", "de kleren"]]
    assert published_answers == ["de kleding", "de kleren"]


def test_pipeline_aligns_partial_audio_with_grouped_answers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "words.txt"
    input_path.write_text(
        "de kleding | de kleren - одежда\n",
        encoding="utf-8",
    )
    settings = make_settings(tmp_path / ".cache", audio_enabled=True)
    captured_audio_by_guid = {}

    def fake_build_deck_package(cards, output_path, deck_name, settings, audio_by_guid=None):  # type: ignore[no-untyped-def]
        captured_audio_by_guid.update(audio_by_guid or {})
        output_path.write_text("complete deck", encoding="utf-8")
        return output_path

    monkeypatch.setattr(pipeline_module, "build_deck_package", fake_build_deck_package)

    result = DeckGenerationPipeline(
        settings,
        llm_client=AlternativeClient(),
        audio_generator=FailingOneAlternativeAudioGenerator(tmp_path / "audio"),
    ).run(input_path=input_path, output_path=tmp_path / "deck.apkg")

    grouped_audio = next(iter(captured_audio_by_guid.values()))
    assert isinstance(grouped_audio, tuple)
    assert len(grouped_audio) == 2
    assert grouped_audio[0] is not None
    assert grouped_audio[0].word_audio is not None
    assert grouped_audio[1] is None
    assert len(result.audio_failed_items) == 2
    assert {failure.source_word for failure in result.audio_failed_items} == {"de kleren"}


def test_pipeline_partial_group_is_cached_but_not_published_until_complete(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "words.txt"
    input_path.write_text(
        "de kleding | de kleren - одежда\n",
        encoding="utf-8",
    )
    output_path = tmp_path / "deck.apkg"
    output_path.write_text("previous complete deck", encoding="utf-8")
    settings = make_settings(tmp_path / ".cache")
    published_answers: list[list[str]] = []

    def fake_build_deck_package(cards, output_path, deck_name, settings, audio_by_guid=None):  # type: ignore[no-untyped-def]
        published_answers.append([source_item.text for source_item, _ in cards])
        output_path.write_text("new complete deck", encoding="utf-8")
        return output_path

    monkeypatch.setattr(pipeline_module, "build_deck_package", fake_build_deck_package)

    failing_client = PartiallyFailingAlternativeClient("de kleren")
    failed_result = DeckGenerationPipeline(settings, llm_client=failing_client).run(
        input_path=input_path,
        output_path=output_path,
    )

    assert failed_result.total_items == 1
    assert failed_result.generated_items == 0
    assert failed_result.cached_items == 0
    assert failed_result.deck_written is False
    assert [item.source_word for item in failed_result.failed_items] == ["de kleren"]
    assert output_path.read_text(encoding="utf-8") == "previous complete deck"
    assert published_answers == []

    retry_client = AlternativeClient()
    retry_result = DeckGenerationPipeline(settings, llm_client=retry_client).run(
        input_path=input_path,
        output_path=output_path,
    )

    assert retry_result.total_items == 1
    assert retry_result.generated_items == 1
    assert retry_result.cached_items == 0
    assert retry_result.deck_written is True
    assert retry_client.pending_batches == [["de kleren"]]
    assert retry_client.existing_contexts == [["de kleding"]]
    assert published_answers == [["de kleding", "de kleren"]]
    assert output_path.read_text(encoding="utf-8") == "new complete deck"


def test_pipeline_counts_grouped_answers_as_one_concept(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    input_path = tmp_path / "words.txt"
    input_path.write_text(
        "de kleding | de kleren - одежда\nleren - учиться\n",
        encoding="utf-8",
    )
    settings = make_settings(tmp_path / ".cache")
    captured_answers: list[str] = []

    def fake_build_deck_package(cards, output_path, deck_name, settings, audio_by_guid=None):  # type: ignore[no-untyped-def]
        captured_answers.extend(source_item.text for source_item, _ in cards)
        output_path.write_text("complete deck", encoding="utf-8")
        return output_path

    monkeypatch.setattr(pipeline_module, "build_deck_package", fake_build_deck_package)

    first_result = DeckGenerationPipeline(settings, llm_client=AlternativeClient()).run(
        input_path=input_path,
        output_path=tmp_path / "first.apkg",
    )

    assert first_result.total_items == 2
    assert first_result.generated_items == 2
    assert first_result.cached_items == 0
    assert captured_answers == ["de kleding", "de kleren", "leren"]

    cached_client = AlternativeClient()
    cached_result = DeckGenerationPipeline(settings, llm_client=cached_client).run(
        input_path=input_path,
        output_path=tmp_path / "cached.apkg",
    )

    assert cached_result.total_items == 2
    assert cached_result.generated_items == 2
    assert cached_result.cached_items == 2
    assert cached_client.calls == 0
