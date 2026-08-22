import json
from pathlib import Path

import pytest

from app.generated_input import load_generated_cards
from app.models import GeneratedCard


def make_phrase_card(word: str) -> GeneratedCard:
    return GeneratedCard(
        dutch_word=word,
        russian_translation="перевод",
        part_of_speech="phrase",
        ipa_transcription="test-ipa",
        lesson_topic="A2",
        form_examples=[
            {
                "kind": "default",
                "form": word,
                "example_sentence_nl": f"Ik gebruik {word}.",
                "example_sentence_ru": "Я использую это выражение.",
            }
        ],
    )


def test_generated_input_rejects_card_associated_with_wrong_word(tmp_path: Path) -> None:
    input_path = tmp_path / "generated-cards.json"
    input_path.write_text(
        json.dumps(
            {
                "format": "dutch-a2-generated-cards",
                "schema_version": 1,
                "concepts": [
                    {
                        "answers": [
                            {
                                "input_item": "leren",
                                "card": make_phrase_card("werken").model_dump(mode="json"),
                            }
                        ]
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="replaced an explicitly accepted Dutch answer"):
        load_generated_cards(input_path)
