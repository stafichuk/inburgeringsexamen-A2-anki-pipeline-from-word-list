"""Domain models and validation rules for generated cards."""

from __future__ import annotations

import re
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    """Base model that strips whitespace and rejects unknown fields."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class PartOfSpeech(str, Enum):
    """Supported part-of-speech categories for the generated cards."""

    NOUN = "noun"
    VERB = "verb"
    ADJECTIVE = "adjective"
    ADVERB = "adverb"
    PRONOUN = "pronoun"
    PREPOSITION = "preposition"
    CONJUNCTION = "conjunction"
    PHRASE = "phrase"
    EXPRESSION = "expression"
    OTHER = "other"


class FormExampleKind(str, Enum):
    """Kinds of form-specific examples rendered on learner cards."""

    SINGULAR = "singular"
    PLURAL = "plural"
    DEFAULT = "default"
    PRESENT_TENSE = "present_tense"
    PAST_TENSE = "past_tense"
    PERFECT_TENSE = "perfect_tense"
    BASE_FORM = "base_form"
    E_FORM = "e_form"
    SINGLE_FORM = "single_form"


def _normalize_enum_value(value: object, enum_type: type[Enum]) -> object:
    """Accept enum values regardless of letter case."""
    if isinstance(value, enum_type):
        return value
    if isinstance(value, str):
        normalized = value.strip().casefold()
        for enum_value in enum_type:
            if isinstance(enum_value.value, str) and enum_value.value.casefold() == normalized:
                return enum_value
    return value


FORM_EXAMPLE_KIND_ORDER = {
    FormExampleKind.SINGULAR: 10,
    FormExampleKind.PLURAL: 20,
    FormExampleKind.DEFAULT: 30,
    FormExampleKind.PRESENT_TENSE: 40,
    FormExampleKind.PAST_TENSE: 50,
    FormExampleKind.PERFECT_TENSE: 60,
    FormExampleKind.BASE_FORM: 70,
    FormExampleKind.E_FORM: 80,
    FormExampleKind.SINGLE_FORM: 90,
}


def _word_tokens(value: str) -> list[str]:
    """Extract normalized word tokens for loose Dutch form matching."""
    return re.findall(r"[\wÀ-ÿ'-]+", value.lower())


def _starts_with_dutch_article(value: str) -> bool:
    """Return whether a Dutch noun-like value starts with a definite article."""
    normalized = value.lower()
    return normalized.startswith("de ") or normalized.startswith("het ")


class FormExample(StrictModel):
    """One example sentence for one visible word form."""

    kind: FormExampleKind
    form: str
    example_sentence_nl: str
    example_sentence_ru: str

    @field_validator("kind", mode="before")
    @classmethod
    def normalize_kind(cls, value: object) -> object:
        """Accept form example kinds regardless of letter case."""
        return _normalize_enum_value(value, FormExampleKind)

    @field_validator("form", "example_sentence_nl", "example_sentence_ru")
    @classmethod
    def ensure_required_text(cls, value: str) -> str:
        """Ensure example fields are non-empty strings."""
        if not value.strip():
            raise ValueError("form example fields must not be empty")
        return value

class VerbForms(StrictModel):
    """Verb forms needed for learner-oriented Dutch cards."""

    infinitive: str
    present_ik: str = Field(description="First-person singular present form, e.g. 'ik leer'.")
    present_hij: str = Field(description="Third-person singular present form, e.g. 'hij leert'.")
    past_tense: str
    perfect_tense: str = Field(description="Compact perfect form, e.g. 'heeft geleerd'.")
    separable_prefix: str | None = None
    conjugation_notes: str | None = None

    @field_validator("infinitive", "present_ik", "present_hij", "past_tense", "perfect_tense")
    @classmethod
    def ensure_required_text(cls, value: str) -> str:
        """Ensure required verb forms are non-empty strings."""
        if not value.strip():
            raise ValueError("verb form fields must not be empty")
        return value

    @field_validator("present_ik")
    @classmethod
    def ensure_present_ik_has_pronoun(cls, value: str) -> str:
        """Require the ik present-tense field to include its pronoun."""
        if re.search(r"\bik\b\s+\S+", value.lower()) is None:
            raise ValueError("present_ik must include an ik form")
        return value

    @field_validator("present_hij")
    @classmethod
    def ensure_present_hij_has_pronoun(cls, value: str) -> str:
        """Require the hij present-tense field to include its pronoun."""
        if re.search(r"\bhij\b\s+\S+", value.lower()) is None:
            raise ValueError("present_hij must include a hij form")
        return value

    @field_validator("separable_prefix", "conjugation_notes")
    @classmethod
    def normalize_optional_text(cls, value: str | None) -> str | None:
        """Reject blank optional strings."""
        if value is not None and not value.strip():
            raise ValueError("optional verb form fields must be null or non-empty")
        return value


class AdjectiveForms(StrictModel):
    """Exceptional adjective note, present only for indeclinable adjectives."""

    onverbuigbaar_example: str = Field(
        description="Short Dutch noun phrase showing the indeclinable adjective, e.g. 'gouden ring'."
    )
    learner_note: str | None = None

    @field_validator("onverbuigbaar_example")
    @classmethod
    def ensure_required_text(cls, value: str) -> str:
        """Ensure the indeclinable adjective example is non-empty."""
        if not value.strip():
            raise ValueError("onverbuigbaar_example must not be empty")
        return value

    @field_validator("learner_note")
    @classmethod
    def normalize_optional_text(cls, value: str | None) -> str | None:
        """Reject blank optional strings."""
        if value is not None and not value.strip():
            raise ValueError("learner_note must be null or non-empty")
        return value


class SourceItem(StrictModel):
    """Input item plus run-time metadata."""

    text: str
    translation_hint: str | None = None
    topic: str | None = None
    lesson: str | None = None
    exam_level: str | None = None

    @field_validator("text")
    @classmethod
    def ensure_text(cls, value: str) -> str:
        """Reject blank source items."""
        if not value.strip():
            raise ValueError("source item text must not be empty")
        return value

    @field_validator("translation_hint")
    @classmethod
    def normalize_translation_hint(cls, value: str | None) -> str | None:
        """Reject blank translation hints."""
        if value is not None and not value.strip():
            raise ValueError("translation_hint must be null or non-empty")
        return value


class GeneratedCard(StrictModel):
    """Strict LLM output schema for one Dutch item."""

    dutch_word: str
    russian_translation: str
    part_of_speech: PartOfSpeech
    ipa_transcription: str
    lesson_topic: str
    form_examples: list[FormExample] = Field(min_length=1, max_length=3)
    tags: list[str] = Field(default_factory=list)
    plural_form: str | None = None
    front_hint: str | None = None
    verb_forms: VerbForms | None = None
    adjective_forms: AdjectiveForms | None = None

    @field_validator("part_of_speech", mode="before")
    @classmethod
    def normalize_part_of_speech(cls, value: object) -> object:
        """Accept part-of-speech values regardless of letter case."""
        return _normalize_enum_value(value, PartOfSpeech)

    @field_validator(
        "dutch_word",
        "russian_translation",
        "ipa_transcription",
        "lesson_topic",
    )
    @classmethod
    def ensure_non_empty(cls, value: str) -> str:
        """Ensure common required string fields are present."""
        if not value.strip():
            raise ValueError("required text fields must not be empty")
        return value

    @field_validator("plural_form", "front_hint")
    @classmethod
    def normalize_optional_common_fields(cls, value: str | None) -> str | None:
        """Reject blank optional noun fields."""
        if value is not None and not value.strip():
            raise ValueError("optional noun fields must be null or non-empty")
        return value

    @field_validator("tags")
    @classmethod
    def normalize_tags(cls, value: list[str]) -> list[str]:
        """Drop blank tags while preserving order."""
        normalized = [tag.strip() for tag in value if tag.strip()]
        return normalized

    @field_validator("form_examples")
    @classmethod
    def validate_unique_form_example_kinds(cls, value: list[FormExample]) -> list[FormExample]:
        """Reject duplicate example kinds so fixed Anki slots stay unambiguous."""
        kinds = [example.kind for example in value]
        if len(kinds) != len(set(kinds)):
            raise ValueError("form_examples must not contain duplicate kinds")
        return value

    def ordered_form_examples(self) -> list[FormExample]:
        """Return form examples in the canonical card-rendering order."""
        return sorted(
            self.form_examples,
            key=lambda example: FORM_EXAMPLE_KIND_ORDER[example.kind],
        )

    @model_validator(mode="after")
    def validate_pos_specific_fields(self) -> "GeneratedCard":
        """Require only the grammar fields relevant to the inferred POS."""
        if self.part_of_speech == PartOfSpeech.NOUN:
            self._require_visible_form_examples()
            if not _starts_with_dutch_article(self.dutch_word):
                raise ValueError("nouns must include article 'de' or 'het' in dutch_word")
            if not self.front_hint:
                raise ValueError("nouns must include front_hint")
            if self.plural_form is None and "множественное число" in self.front_hint:
                raise ValueError("uncountable nouns must not prompt plural recall")
            if self.verb_forms is not None:
                raise ValueError("nouns must not include verb_forms")
            if self.adjective_forms is not None:
                raise ValueError("nouns must not include adjective_forms")
            if self.plural_form is None:
                self._normalize_uncountable_form_example_kind()
                self._require_exact_example_kinds({FormExampleKind.DEFAULT}, "uncountable nouns")
            else:
                if _starts_with_dutch_article(self.plural_form):
                    raise ValueError("countable noun plural_form must not include article")
                self._require_exact_example_kinds(
                    {FormExampleKind.SINGULAR, FormExampleKind.PLURAL},
                    "countable nouns",
                )
            return self

        if self.part_of_speech == PartOfSpeech.VERB:
            if self.verb_forms is None:
                raise ValueError("verbs must include verb_forms")
            if self.plural_form or self.front_hint:
                raise ValueError("verbs must not include noun-only fields")
            if self.adjective_forms is not None:
                raise ValueError("verbs must not include adjective_forms")
            self._require_exact_example_kinds(
                {
                    FormExampleKind.PRESENT_TENSE,
                    FormExampleKind.PAST_TENSE,
                    FormExampleKind.PERFECT_TENSE,
                },
                "verbs",
            )
            return self

        if self.part_of_speech == PartOfSpeech.ADJECTIVE:
            self._require_visible_form_examples()
            if self.plural_form or self.front_hint:
                raise ValueError("adjectives must not include noun-only fields")
            if self.verb_forms is not None:
                raise ValueError("adjectives must not include verb_forms")
            example_kinds = {example.kind for example in self.form_examples}
            if example_kinds == {FormExampleKind.BASE_FORM, FormExampleKind.E_FORM}:
                if self.adjective_forms is not None:
                    raise ValueError("regular adjectives must not include adjective_forms")
                return self
            if example_kinds == {FormExampleKind.SINGLE_FORM}:
                return self
            raise ValueError(
                "adjectives must include either base_form/e_form examples "
                "or one single_form example"
            )

        if self.plural_form or self.front_hint:
            raise ValueError("non-nouns must not include noun-only fields")
        if self.verb_forms is not None:
            raise ValueError("non-verbs must not include verb_forms")
        if self.adjective_forms is not None:
            raise ValueError("non-adjectives must not include adjective_forms")
        self._require_visible_form_examples()
        self._require_exact_example_kinds({FormExampleKind.DEFAULT}, "single-form words")
        return self

    def _require_visible_form_examples(self) -> None:
        """Require selected forms to be visible for non-verb examples."""
        for example in self.form_examples:
            form_tokens = set(_word_tokens(example.form))
            example_tokens = set(_word_tokens(example.example_sentence_nl))
            if form_tokens and not form_tokens.issubset(example_tokens):
                raise ValueError("form must appear in example_sentence_nl")

    def _normalize_uncountable_form_example_kind(self) -> None:
        """Treat one singular example as the default example for uncountable nouns."""
        if len(self.form_examples) != 1:
            return
        example = self.form_examples[0]
        if example.kind == FormExampleKind.SINGULAR:
            self.form_examples = [example.model_copy(update={"kind": FormExampleKind.DEFAULT})]

    def _require_exact_example_kinds(self, required_kinds: set[FormExampleKind], label: str) -> None:
        """Require a POS-specific exact set of example kinds."""
        actual_kinds = {example.kind for example in self.form_examples}
        if actual_kinds == required_kinds:
            return

        missing = sorted(kind.value for kind in required_kinds - actual_kinds)
        unexpected = sorted(kind.value for kind in actual_kinds - required_kinds)
        details: list[str] = []
        if missing:
            details.append(f"missing {', '.join(missing)}")
        if unexpected:
            details.append(f"unexpected {', '.join(unexpected)}")
        raise ValueError(f"{label} must include exact form_examples kinds: {'; '.join(details)}")
