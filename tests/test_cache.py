from pathlib import Path

from app.cache import CardCache
from app.models import SourceItem


def test_cache_key_includes_translation_hint(tmp_path: Path) -> None:
    cache = CardCache(tmp_path)
    nephew = SourceItem(text="de neef", translation_hint="племянник", topic="Familie", lesson="Les 1")
    cousin = SourceItem(text="de neef", translation_hint="двоюродный брат", topic="Familie", lesson="Les 1")
    plain = SourceItem(text="de neef", topic="Familie", lesson="Les 1")

    nephew_key = cache.build_key(nephew, model_name="test-model", prompt_version="test-prompt")
    cousin_key = cache.build_key(cousin, model_name="test-model", prompt_version="test-prompt")
    plain_key = cache.build_key(plain, model_name="test-model", prompt_version="test-prompt")

    assert nephew_key != cousin_key
    assert nephew_key != plain_key
    assert cousin_key != plain_key
