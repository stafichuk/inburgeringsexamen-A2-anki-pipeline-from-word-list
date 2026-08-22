import json
from pathlib import Path
import zipfile

import pytest

from app.cli import main


def test_cli_assemble_mode_builds_from_generated_input_without_llm_config(
    tmp_path: Path,
) -> None:
    generated_input_path = tmp_path / "generated-cards.json"
    generated_input_path.write_text(
        json.dumps(
            {
                "format": "dutch-a2-generated-cards",
                "schema_version": 1,
                "concepts": [
                    {
                        "entry_id": "yesterday",
                        "translation_hint": "вчера",
                        "topic": "Tijd",
                        "lesson": "Les 1",
                        "exam_level": "A2",
                        "answers": [
                            {
                                "input_item": "gisteren",
                                "card": {
                                    "dutch_word": "gisteren",
                                    "russian_translation": "вчера",
                                    "part_of_speech": "adverb",
                                    "ipa_transcription": "ˈɣɪstərə(n)",
                                    "lesson_topic": "Tijd",
                                    "form_examples": [
                                        {
                                            "kind": "default",
                                            "form": "gisteren",
                                            "example_sentence_nl": "Gisteren werkte ik thuis.",
                                            "example_sentence_ru": "Вчера я работал дома.",
                                        }
                                    ],
                                    "tags": ["tijd"],
                                },
                            }
                        ],
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text('deck:\n  deck_name: "Offline deck"\n', encoding="utf-8")
    output_path = tmp_path / "deck.apkg"

    exit_code = main(
        [
            "--mode",
            "assemble",
            "--input",
            str(generated_input_path),
            "--output",
            str(output_path),
            "--config",
            str(config_path),
        ]
    )

    assert exit_code == 0
    with zipfile.ZipFile(output_path) as package:
        assert {"collection.anki2", "media"}.issubset(package.namelist())


@pytest.mark.parametrize(
    "generation_only_option",
    [
        ["--force"],
        ["--model", "test-model"],
        ["--timeout", "0"],
    ],
)
def test_cli_assemble_mode_rejects_generation_only_options(
    generation_only_option: list[str],
    capsys,
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "--mode",
                "assemble",
                "--input",
                "generated-cards.json",
                "--output",
                "deck.apkg",
                *generation_only_option,
            ]
        )

    assert exc_info.value.code == 2
    assert "assemble mode does not accept generation-only option" in capsys.readouterr().err
