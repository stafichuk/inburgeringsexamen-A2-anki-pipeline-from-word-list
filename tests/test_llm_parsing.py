import json

import pytest

from app.config import LLMSettings
from app.llm_client import (
    LLMClient,
    LLMResponseFormatError,
    extract_json_object,
    parse_generated_card,
    parse_generated_cards,
)
from app.models import GeneratedCard, SourceConcept, SourceItem


VALID_JSON = """
{
  "dutch_word": "leren",
  "russian_translation": "учиться",
  "part_of_speech": "verb",
  "ipa_transcription": "ˈleːrə(n)",
  "lesson_topic": "De school",
  "form_examples": [
    {
      "kind": "present_tense",
      "form": "ik leer",
      "example_sentence_nl": "Ik leer Nederlands op school.",
      "example_sentence_ru": "Я учу нидерландский в школе."
    },
    {
      "kind": "past_tense",
      "form": "leerde",
      "example_sentence_nl": "Ik leerde gisteren nieuwe woorden.",
      "example_sentence_ru": "Вчера я учил новые слова."
    },
    {
      "kind": "perfect_tense",
      "form": "heeft geleerd",
      "example_sentence_nl": "Hij heeft Nederlands geleerd.",
      "example_sentence_ru": "Он выучил нидерландский."
    }
  ],
  "tags": ["school", "verb"],
  "plural_form": null,
  "front_hint": null,
  "verb_forms": {
    "infinitive": "leren",
    "present_ik": "ik leer",
    "present_hij": "hij leert",
    "past_tense": "leerde",
    "perfect_tense": "heeft geleerd",
    "separable_prefix": null,
    "conjugation_notes": "regular weak verb"
  },
  "adjective_forms": null
}
""".strip()

INVALID_MONTH_JSON = """
{
  "dutch_word": "januari",
  "russian_translation": "январь",
  "part_of_speech": "noun",
  "ipa_transcription": "/jɑnyˈwaːri/",
  "lesson_topic": "Het formulier en de agenda",
  "form_examples": [
    {
      "kind": "default",
      "form": "januari",
      "example_sentence_nl": "Januari is de eerste maand van het jaar.",
      "example_sentence_ru": "Январь — первый месяц года."
    }
  ],
  "tags": ["maand", "kalender", "agenda", "A2"],
  "plural_form": null,
  "front_hint": "месяц (январь)",
  "verb_forms": null,
  "adjective_forms": null
}
""".strip()

VALID_MONTH_JSON = """
{
  "dutch_word": "de januari",
  "russian_translation": "январь",
  "part_of_speech": "noun",
  "ipa_transcription": "/jɑnyˈwaːri/",
  "lesson_topic": "Het formulier en de agenda",
  "form_examples": [
    {
      "kind": "default",
      "form": "januari",
      "example_sentence_nl": "Januari is de eerste maand van het jaar.",
      "example_sentence_ru": "Январь — первый месяц года."
    }
  ],
  "tags": ["maand", "kalender", "agenda", "A2"],
  "plural_form": null,
  "front_hint": "месяц (январь)",
  "verb_forms": null,
  "adjective_forms": null
}
""".strip()


def _card_payload(raw_json: str) -> dict[str, object]:
    payload = json.loads(raw_json)
    assert isinstance(payload, dict)
    return payload


def _batch_response(*entries: dict[str, object]) -> str:
    return json.dumps({"cards": list(entries)}, ensure_ascii=False)


def _batch_entry(
    source_id: int,
    input_item: str,
    raw_card: str,
    *,
    translation_hint: str | None = None,
) -> dict[str, object]:
    return {
        "source_id": source_id,
        "input_item": input_item,
        "translation_hint": translation_hint,
        "card": _card_payload(raw_card),
    }


def _accepted_adverb_card() -> GeneratedCard:
    return GeneratedCard.model_validate(
        {
            "dutch_word": "gisteren",
            "russian_translation": "вчера",
            "part_of_speech": "adverb",
            "ipa_transcription": "/ˈɣɪstərə(n)/",
            "lesson_topic": "Tijd",
            "form_examples": [
                {
                    "kind": "default",
                    "form": "gisteren",
                    "example_sentence_nl": "Gisteren fietste Noor naar haar werk.",
                    "example_sentence_ru": "Вчера Нур ездила на работу на велосипеде.",
                }
            ],
            "tags": ["tijd", "adverb"],
            "plural_form": None,
            "front_hint": None,
            "verb_forms": None,
            "adjective_forms": None,
        }
    )


def test_extract_json_object_handles_fenced_response() -> None:
    raw_text = f"```json\n{VALID_JSON}\n```"
    extracted = extract_json_object(raw_text)
    assert extracted.startswith("{")
    assert extracted.endswith("}")


def test_parse_generated_card_accepts_valid_json() -> None:
    card = parse_generated_card(VALID_JSON)
    assert card.verb_forms is not None
    assert card.verb_forms.perfect_tense == "heeft geleerd"
    assert len(card.form_examples) == 3


def test_parse_generated_card_rejects_invalid_payload() -> None:
    broken_json = '{"dutch_word": "leren"}'
    with pytest.raises(LLMResponseFormatError) as exc_info:
        parse_generated_card(broken_json)

    assert exc_info.value.raw_text == broken_json
    assert "LLM response:" in str(exc_info.value)
    assert broken_json in str(exc_info.value)


def test_parse_generated_cards_keeps_valid_cards_from_partially_invalid_batch() -> None:
    raw_text = _batch_response(
        _batch_entry(42, "leren", VALID_JSON),
        _batch_entry(7, "januari", INVALID_MONTH_JSON),
    )

    result = parse_generated_cards(
        raw_text,
        expected_items={
            7: SourceItem(text="januari"),
            42: SourceItem(text="leren"),
        },
    )

    assert set(result.cards) == {42}
    assert result.cards[42].dutch_word == "leren"
    assert set(result.errors) == {7}
    assert "nouns must include article" in result.errors[7]


def test_parse_generated_cards_matches_reversed_output_by_source_id() -> None:
    raw_text = _batch_response(
        _batch_entry(2, "januari", VALID_MONTH_JSON),
        _batch_entry(1, "leren", VALID_JSON),
    )

    result = parse_generated_cards(
        raw_text,
        expected_items={
            1: SourceItem(text="leren"),
            2: SourceItem(text="januari"),
        },
    )

    assert result.errors == {}
    assert result.cards[1].dutch_word == "leren"
    assert result.cards[2].dutch_word == "de januari"


def test_parse_generated_cards_rejects_swapped_input_item_echoes() -> None:
    raw_text = _batch_response(
        _batch_entry(1, "januari", VALID_JSON),
        _batch_entry(2, "leren", VALID_MONTH_JSON),
    )

    result = parse_generated_cards(
        raw_text,
        expected_items={
            1: SourceItem(text="leren"),
            2: SourceItem(text="januari"),
        },
    )

    assert result.cards == {}
    assert "expected 'leren', got 'januari'" in result.errors[1]
    assert "expected 'januari', got 'leren'" in result.errors[2]


def test_parse_generated_cards_rejects_swapped_hints_for_duplicate_input_items() -> None:
    raw_text = _batch_response(
        _batch_entry(
            1,
            "de neef",
            VALID_JSON,
            translation_hint="двоюродный брат",
        ),
        _batch_entry(
            2,
            "de neef",
            VALID_MONTH_JSON,
            translation_hint="племянник",
        ),
    )

    result = parse_generated_cards(
        raw_text,
        expected_items={
            1: SourceItem(text="de neef", translation_hint="племянник"),
            2: SourceItem(text="de neef", translation_hint="двоюродный брат"),
        },
    )

    assert result.cards == {}
    assert "expected 'племянник', got 'двоюродный брат'" in result.errors[1]
    assert "expected 'двоюродный брат', got 'племянник'" in result.errors[2]


def test_parse_generated_cards_requires_explicit_null_translation_hint() -> None:
    entry = _batch_entry(1, "leren", VALID_JSON)
    del entry["translation_hint"]

    result = parse_generated_cards(
        _batch_response(entry),
        expected_items={1: SourceItem(text="leren")},
    )

    assert result.cards == {}
    assert result.errors == {1: "translation_hint is missing from the response wrapper"}


def test_parse_generated_cards_accepts_exact_explicit_group_answer() -> None:
    concept = SourceConcept(
        dutch_answers=("de kleding", "de kleren"),
        translation_hint="одежда",
    )
    card_payload = _card_payload(VALID_MONTH_JSON)
    card_payload.update(
        {
            "dutch_word": "de kleding",
            "russian_translation": "одежда",
            "form_examples": [
                {
                    "kind": "default",
                    "form": "kleding",
                    "example_sentence_nl": "Ik koop nieuwe kleding.",
                    "example_sentence_ru": "Я покупаю новую одежду.",
                }
            ],
            "front_hint": "одежда",
        }
    )

    result = parse_generated_cards(
        _batch_response(
            {
                "source_id": 1,
                "input_item": "de kleding",
                "translation_hint": "одежда",
                "card": card_payload,
            }
        ),
        expected_items={1: concept.source_items()[0]},
    )

    assert result.errors == {}
    assert result.cards[1].dutch_word == "de kleding"


def test_parse_generated_cards_rejects_unlisted_group_synonym() -> None:
    concept = SourceConcept(
        dutch_answers=("de kleding", "de kleren"),
        translation_hint="одежда",
    )

    result = parse_generated_cards(
        _batch_response(
            _batch_entry(
                1,
                "de kleding",
                VALID_MONTH_JSON,
                translation_hint="одежда",
            )
        ),
        expected_items={1: concept.source_items()[0]},
    )

    assert result.cards == {}
    assert "replaced an explicitly accepted Dutch answer" in result.errors[1]
    assert "expected 'de kleding', got 'de januari'" in result.errors[1]


def test_parse_generated_cards_reports_missing_duplicate_and_unknown_ids() -> None:
    raw_text = _batch_response(
        _batch_entry(2, "leren", VALID_JSON),
        _batch_entry(99, "januari", VALID_MONTH_JSON),
        _batch_entry(2, "leren", VALID_MONTH_JSON),
    )

    result = parse_generated_cards(
        raw_text,
        expected_items={
            1: SourceItem(text="de school"),
            2: SourceItem(text="leren"),
            3: SourceItem(text="de vriend"),
        },
    )

    assert result.cards == {}
    assert result.errors == {
        1: "missing from response",
        2: "returned more than once",
        3: "missing from response",
    }
    assert "Ignored unknown source_id 99" in result.global_errors
    feedback = result.validation_feedback()
    assert "source_id 1: missing from response" in feedback
    assert "source_id 2: returned more than once" in feedback
    assert "source_id 3: missing from response" in feedback
    assert "Ignored unknown source_id 99" in feedback


def test_generate_cards_retries_only_unresolved_and_accepts_cards_before_backoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class RepairingBatchClient(LLMClient):
        def __init__(self, settings: LLMSettings) -> None:
            super().__init__(settings)
            self.calls: list[dict[str, object]] = []

        def _request_completion(
            self,
            pending_items: list[tuple[int, SourceItem]],
            existing_cards: list[tuple[SourceItem, GeneratedCard]],
            *,
            validation_error: str | None = None,
        ) -> str:
            pending_ids = [source_id for source_id, _ in pending_items]
            events.append(f"request:{pending_ids}")
            self.calls.append(
                {
                    "pending_ids": pending_ids,
                    "context": [
                        (source_item.text, card.dutch_word)
                        for source_item, card in existing_cards
                    ],
                    "validation_error": validation_error,
                }
            )
            if len(self.calls) == 1:
                return _batch_response(
                    _batch_entry(202, "leren", VALID_JSON),
                    _batch_entry(101, "januari", INVALID_MONTH_JSON),
                )
            return _batch_response(_batch_entry(101, "januari", VALID_MONTH_JSON))

    settings = LLMSettings(
        base_url="https://example.invalid/v1/chat/completions",
        api_token="token",
        model_name="test-model",
        max_retries=1,
        retry_backoff_seconds=0,
    )
    client = RepairingBatchClient(settings)
    accepted_context = [(SourceItem(text="gisteren"), _accepted_adverb_card())]
    monkeypatch.setattr(
        "app.llm_client.time.sleep",
        lambda _: events.append("backoff"),
    )

    result = client.generate_cards(
        [
            (101, SourceItem(text="januari")),
            (202, SourceItem(text="leren")),
        ],
        accepted_context,
        on_card_accepted=lambda source_id, card: events.append(
            f"accepted:{source_id}:{card.dutch_word}"
        ),
    )

    assert set(result.cards) == {101, 202}
    assert result.failures == {}
    assert client.calls[0] == {
        "pending_ids": [101, 202],
        "context": [("gisteren", "gisteren")],
        "validation_error": None,
    }
    assert client.calls[1]["pending_ids"] == [101]
    assert client.calls[1]["context"] == [
        ("gisteren", "gisteren"),
        ("leren", "leren"),
    ]
    validation_error = client.calls[1]["validation_error"]
    assert isinstance(validation_error, str)
    assert "source_id 101" in validation_error
    assert "nouns must include article" in validation_error
    assert events == [
        "request:[101, 202]",
        "accepted:202:leren",
        "backoff",
        "request:[101]",
        "accepted:101:de januari",
    ]
