from app.models import SourceItem
from app.prompts import build_messages


def test_prompt_passes_translation_hint_as_strict_sense_constraint() -> None:
    source_item = SourceItem(text="de neef", translation_hint="племянник", topic="Familie")

    messages = build_messages(source_item)
    user_prompt = messages[1]["content"]

    assert "Input item:\nde neef" in user_prompt
    assert "Translation hint:\nплемянник" in user_prompt
    assert "strict sense constraint" in user_prompt
    assert "alternative meanings" in user_prompt
    assert "de neef - племянник" not in user_prompt
