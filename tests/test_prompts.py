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

`
def test_prompt_teaches_month_names_as_de_nouns() -> None:
    source_item = SourceItem(text="januari", topic="Het formulier en de agenda")

    messages = build_messages(source_item)
    user_prompt = messages[1]["content"]

    assert "Month names are Dutch de-nouns" in user_prompt
    assert '"de januari"' in user_prompt
    assert '"de december"' in user_prompt
    assert "Use the bare month name in form_examples" in user_prompt


def test_prompt_includes_validation_feedback_for_repair_attempt() -> None:
    previous_response = '{"dutch_word": "januari"}'
    validation_error = "nouns must include article 'de' or 'het' in dutch_word"
    source_item = SourceItem(text="januari", topic="Het formulier en de agenda")

    messages = build_messages(
        source_item,
        previous_response=previous_response,
        validation_error=validation_error,
    )

    assert len(messages) == 3
    repair_prompt = messages[2]["content"]
    assert "previous response failed validation" in repair_prompt
    assert validation_error in repair_prompt
    assert previous_response in repair_prompt
