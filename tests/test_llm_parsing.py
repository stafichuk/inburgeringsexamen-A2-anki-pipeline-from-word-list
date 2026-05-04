import pytest

from app.llm_client import LLMResponseFormatError, extract_json_object, parse_generated_card


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
      "form": "leer",
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
      "kind": "past_participle",
      "form": "geleerd",
      "example_sentence_nl": "Ik heb Nederlands geleerd.",
      "example_sentence_ru": "Я выучил нидерландский."
    }
  ],
  "tags": ["school", "verb"],
  "plural_form": null,
  "front_hint": null,
  "verb_forms": {
    "infinitive": "leren",
    "present_tense": "ik leer, jij leert, hij leert",
    "past_tense": "leerde, leerden",
    "past_participle": "geleerd",
    "perfect_example": "Ik heb Nederlands geleerd.",
    "separable_prefix": null,
    "conjugation_notes": "regular weak verb"
  },
  "adjective_forms": null
}
""".strip()


def test_extract_json_object_handles_fenced_response() -> None:
    raw_text = f"```json\n{VALID_JSON}\n```"
    extracted = extract_json_object(raw_text)
    assert extracted.startswith("{")
    assert extracted.endswith("}")


def test_parse_generated_card_accepts_valid_json() -> None:
    card = parse_generated_card(VALID_JSON)
    assert card.verb_forms is not None
    assert card.verb_forms.past_participle == "geleerd"
    assert len(card.form_examples) == 3


def test_parse_generated_card_rejects_invalid_payload() -> None:
    broken_json = '{"dutch_word": "leren"}'
    with pytest.raises(LLMResponseFormatError):
        parse_generated_card(broken_json)
