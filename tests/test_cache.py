import json
from pathlib import Path

from app.cache import CardCache
from app.models import GeneratedCard, SourceItem


def make_card() -> GeneratedCard:
    return GeneratedCard(
        dutch_word="gisteren",
        russian_translation="вчера",
        part_of_speech="adverb",
        ipa_transcription="ˈɣɪstərə(n)",
        lesson_topic="Tijd",
        form_examples=[
            {
                "kind": "default",
                "form": "gisteren",
                "example_sentence_nl": "Gisteren werkte ik thuis.",
                "example_sentence_ru": "Вчера я работал дома.",
            }
        ],
        tags=["tijd", "adverb"],
    )


def test_cache_key_includes_translation_hint(tmp_path: Path) -> None:
    cache = CardCache(tmp_path)
    nephew = SourceItem(text="de neef", translation_hint="племянник", topic="Familie", lesson="Les 1")
    cousin = SourceItem(text="de neef", translation_hint="двоюродный брат", topic="Familie", lesson="Les 1")
    plain = SourceItem(text="de neef", topic="Familie", lesson="Les 1")

    nephew_key = cache.build_key(nephew)
    cousin_key = cache.build_key(cousin)
    plain_key = cache.build_key(plain)

    assert nephew_key != cousin_key
    assert nephew_key != plain_key
    assert cousin_key != plain_key


def test_cache_key_includes_learning_context(tmp_path: Path) -> None:
    cache = CardCache(tmp_path)
    original = SourceItem(text="de vriend", topic="Familie", lesson="Les 1", exam_level="A2")
    other_topic = original.model_copy(update={"topic": "Werk"})
    other_lesson = original.model_copy(update={"lesson": "Les 2"})
    other_level = original.model_copy(update={"exam_level": "B1"})

    assert cache.build_key(original) == cache.build_key(original)
    assert cache.build_key(original) != cache.build_key(other_topic)
    assert cache.build_key(original) != cache.build_key(other_lesson)
    assert cache.build_key(original) != cache.build_key(other_level)


def test_model_and_prompt_are_provenance_but_not_cache_identity(tmp_path: Path) -> None:
    cache = CardCache(tmp_path)
    source_item = SourceItem(text="gisteren", topic="Tijd", lesson="Les 1", exam_level="A2")
    card = make_card()

    cache.set(source_item, card, model_name="model-a", prompt_version="prompt-a")
    first_path = next(tmp_path.glob("*.json"))
    assert cache.get(source_item) == card

    cache.set(source_item, card, model_name="model-b", prompt_version="prompt-b")

    assert list(tmp_path.glob("*.json")) == [first_path]
    payload = json.loads(first_path.read_text(encoding="utf-8"))
    assert payload["model_name"] == "model-b"
    assert payload["prompt_version"] == "prompt-b"


def test_explicit_id_does_not_hide_source_corrections_from_cache(tmp_path: Path) -> None:
    cache = CardCache(tmp_path)
    original = SourceItem(entry_id="friend", text="de vrient", topic="Familie", lesson="Les 1")
    corrected = SourceItem(entry_id="friend", text="de vriend", topic="Familie", lesson="Les 1")

    assert original.identity_key() == corrected.identity_key()
    assert cache.build_key(original) != cache.build_key(corrected)
