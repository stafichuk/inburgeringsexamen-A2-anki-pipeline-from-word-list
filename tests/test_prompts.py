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


def test_prompt_normalizes_plural_noun_inputs_to_singular_lemma() -> None:
    source_item = SourceItem(text="de ouders", topic="Familie")

    messages = build_messages(source_item)
    user_prompt = messages[1]["content"]

    assert "Input item:\nde ouders" in user_prompt
    assert "already a plural noun" in user_prompt
    assert "de ouder" in user_prompt
    assert "ouders" in user_prompt
