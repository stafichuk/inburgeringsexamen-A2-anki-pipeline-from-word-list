import time
from pathlib import Path

import app.pipeline as pipeline_module
from app.cache import CardCache
from app.config import AppSettings
from app.models import GeneratedCard, SourceItem, VerbForms
from app.pipeline import DeckGenerationPipeline


def make_settings(cache_dir: Path, *, parallelism: int = 4) -> AppSettings:
    return AppSettings.model_validate(
        {
            "llm": {
                "base_url": "https://example.invalid/v1/chat/completions",
                "api_token": "token",
                "model_name": "test-model",
            },
            "generation": {"parallelism": parallelism},
            "cache": {"directory": str(cache_dir)},
        }
    )


def make_card(word: str) -> GeneratedCard:
    return GeneratedCard(
        dutch_word=word,
        russian_translation="учиться",
        part_of_speech="verb",
        ipa_transcription="ˈleːrə(n)",
        example_sentence_nl=f"Ik gebruik {word} in de les.",
        example_sentence_ru=f"Я использую {word} на уроке.",
        lesson_topic="De school",
        tags=["school", "verb"],
        verb_forms=VerbForms(
            infinitive=word,
            present_tense=f"ik {word}, jij {word}t, hij {word}t",
            past_tense="leerde, leerden",
            past_participle="geleerd",
        ),
    )


class FakeClient:
    def __init__(self) -> None:
        self.calls = 0

    def generate_card(self, source_item: SourceItem) -> GeneratedCard:
        self.calls += 1
        return make_card(source_item.text)


class PartiallyFailingClient(FakeClient):
    def generate_card(self, source_item: SourceItem) -> GeneratedCard:
        self.calls += 1
        if source_item.text == "fout":
            raise ValueError("invalid payload")
        return make_card(source_item.text)


class SlowOrderedClient(FakeClient):
    def generate_card(self, source_item: SourceItem) -> GeneratedCard:
        self.calls += 1
        delays = {
            "eerste": 0.2,
            "tweede": 0.05,
            "derde": 0.1,
        }
        time.sleep(delays[source_item.text])
        return make_card(source_item.text)


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

    second_client = FakeClient()
    second_pipeline = DeckGenerationPipeline(settings, llm_client=second_client, cache=CardCache(settings.cache.directory))
    second_result = second_pipeline.run(input_path=input_path, output_path=tmp_path / "deck-2.apkg")

    assert second_result.cached_items == 1
    assert second_client.calls == 0


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


def test_pipeline_writes_partial_deck_and_reports_failures(tmp_path: Path) -> None:
    input_path = tmp_path / "words.txt"
    input_path.write_text("leren\nfout\n", encoding="utf-8")
    output_path = tmp_path / "deck.apkg"
    settings = make_settings(tmp_path / ".cache")

    client = PartiallyFailingClient()
    result = DeckGenerationPipeline(settings, llm_client=client).run(input_path=input_path, output_path=output_path)

    assert output_path.exists()
    assert result.generated_items == 1
    assert len(result.failed_items) == 1
    assert result.failed_items[0].source_word == "fout"


def test_pipeline_preserves_input_order_with_parallel_generation(tmp_path: Path, monkeypatch) -> None:
    input_path = tmp_path / "words.txt"
    input_path.write_text("eerste\ntweede\nderde\n", encoding="utf-8")
    output_path = tmp_path / "deck.apkg"
    settings = make_settings(tmp_path / ".cache", parallelism=3)
    captured_order: list[str] = []

    def fake_build_deck_package(cards, output_path, deck_name, settings):  # type: ignore[no-untyped-def]
        captured_order.extend(source_item.text for source_item, _ in cards)
        output_path.write_text("stub", encoding="utf-8")
        return output_path

    monkeypatch.setattr(pipeline_module, "build_deck_package", fake_build_deck_package)

    client = SlowOrderedClient()
    result = DeckGenerationPipeline(settings, llm_client=client).run(input_path=input_path, output_path=output_path)

    assert result.generated_items == 3
    assert captured_order == ["eerste", "tweede", "derde"]
