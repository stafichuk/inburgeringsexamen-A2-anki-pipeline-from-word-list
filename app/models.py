"""Domain models and validation rules for generated cards."""

from __future__ import annotations

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


class VerbForms(StrictModel):
    """Verb forms needed for learner-oriented Dutch cards."""

    infinitive: str
    present_tense: str
    past_tense: str
    past_participle: str
    perfect_example: str | None = None
    separable_prefix: str | None = None
    conjugation_notes: str | None = None

    @field_validator("infinitive", "present_tense", "past_tense", "past_participle")
    @classmethod
    def ensure_required_text(cls, value: str) -> str:
        """Ensure required verb forms are non-empty strings."""
        if not value.strip():
            raise ValueError("verb form fields must not be empty")
        return value

    @field_validator("perfect_example", "separable_prefix", "conjugation_notes")
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


class GeneratedCard(StrictModel):
    """Strict LLM output schema for one Dutch item."""

    dutch_word: str
    russian_translation: str
    part_of_speech: PartOfSpeech
    ipa_transcription: str
    example_sentence_nl: str
    example_sentence_ru: str
    lesson_topic: str
    tags: list[str] = Field(default_factory=list)
    article: str | None = None
    plural_form: str | None = None
    front_hint: str | None = None
    verb_forms: VerbForms | None = None
    adjective_forms: AdjectiveForms | None = None

    @field_validator(
        "dutch_word",
        "russian_translation",
        "ipa_transcription",
        "example_sentence_nl",
        "example_sentence_ru",
        "lesson_topic",
    )
    @classmethod
    def ensure_non_empty(cls, value: str) -> str:
        """Ensure common required string fields are present."""
        if not value.strip():
            raise ValueError("required text fields must not be empty")
        return value

    @field_validator("article", "plural_form", "front_hint")
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

    @model_validator(mode="after")
    def validate_pos_specific_fields(self) -> "GeneratedCard":
        """Require only the grammar fields relevant to the inferred POS."""
        if self.part_of_speech == PartOfSpeech.NOUN:
            if self.article not in {"de", "het"}:
                raise ValueError("nouns must include article 'de' or 'het'")
            if not self.front_hint:
                raise ValueError("nouns must include front_hint")
            if self.plural_form is None and "множественное число" in self.front_hint:
                raise ValueError("uncountable nouns must not prompt plural recall")
            if self.verb_forms is not None:
                raise ValueError("nouns must not include verb_forms")
            if self.adjective_forms is not None:
                raise ValueError("nouns must not include adjective_forms")
            return self

        if self.part_of_speech == PartOfSpeech.VERB:
            if self.verb_forms is None:
                raise ValueError("verbs must include verb_forms")
            if self.article or self.plural_form or self.front_hint:
                raise ValueError("verbs must not include noun-only fields")
            if self.adjective_forms is not None:
                raise ValueError("verbs must not include adjective_forms")
            return self

        if self.part_of_speech == PartOfSpeech.ADJECTIVE:
            if self.article or self.plural_form or self.front_hint:
                raise ValueError("adjectives must not include noun-only fields")
            if self.verb_forms is not None:
                raise ValueError("adjectives must not include verb_forms")
            return self

        if self.article or self.plural_form or self.front_hint:
            raise ValueError("non-nouns must not include noun-only fields")
        if self.verb_forms is not None:
            raise ValueError("non-verbs must not include verb_forms")
        if self.adjective_forms is not None:
            raise ValueError("non-adjectives must not include adjective_forms")
        return self
