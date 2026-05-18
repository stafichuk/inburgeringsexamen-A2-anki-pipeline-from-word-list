import time
from pathlib import Path

import app.pipeline as pipeline_module
from app.audio import AudioGenerationError
from app.cache import CardCache
from app.config import AppSettings
from app.models import GeneratedCard, SourceItem, VerbForms
from app.pipeline import DeckGenerationPipeline, load_source_items


def make_settings(cache_dir: Path, *, parallelism: int = 4, audio_enabled: bool = False) -> AppSettings:
    payload = {
        "llm": {
            "base_url": "https://example.invalid/v1/chat/completions",
            "api_token": "token",
            "model_name": "test-model",
        },
        "generation": {"parallelism": parallelism},
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


class FakeClient:
    def __init__(self) -> None:
        self.calls = 0

    def generate_card(self, source_item: SourceItem) -> GeneratedCard:
        self.calls += 1
        return make_card(source_item.text)


class NounClient(FakeClient):
    def generate_card(self, source_item: SourceItem) -> GeneratedCard:
        self.calls += 1
        return make_noun_card(source_item.text)


class TranslationHintNounClient(FakeClient):
    def generate_card(self, source_item: SourceItem) -> GeneratedCard:
        self.calls += 1
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

    def fake_build_deck_package(cards, output_path, deck_name, settings, audio_by_guid=None):  # type: ignore[no-untyped-def]
        captured_order.extend(source_item.text for source_item, _ in cards)
        output_path.write_text("stub", encoding="utf-8")
        return output_path

    monkeypatch.setattr(pipeline_module, "build_deck_package", fake_build_deck_package)

    client = SlowOrderedClient()
    result = DeckGenerationPipeline(settings, llm_client=client).run(input_path=input_path, output_path=output_path)

    assert result.generated_items == 3
    assert captured_order == ["eerste", "tweede", "derde"]


def test_pipeline_keeps_duplicate_words_with_distinct_translation_hints(tmp_path: Path, monkeypatch) -> None:
    input_path = tmp_path / "words.txt"
    input_path.write_text("de neef - племянник\nde neef - двоюродный брат\n", encoding="utf-8")
    output_path = tmp_path / "deck.apkg"
    settings = make_settings(tmp_path / ".cache", parallelism=1)
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
    assert client.calls == 2
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
                "de neef - племянник",
                "de neef - двоюродный брат",
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
    assert all(item.topic == "Familie" for item in items)


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
