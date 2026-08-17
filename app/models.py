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


class NounNumber(str, Enum):
    """Number behavior relevant to noun generation and card rendering."""

    COUNTABLE = "countable"
    UNCOUNTABLE = "uncountable"
    PLURAL_ONLY = "plural_only"


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
    normalized = value.strip().casefold()
    return normalized.startswith("de ") or normalized.startswith("het ")


def _strip_dutch_article(value: str) -> str:
    """Return a Dutch noun-like value without a leading definite article."""
    stripped = value.strip()
    normalized = stripped.casefold()
    for article in ("de ", "het "):
        if normalized.startswith(article):
            return stripped[len(article) :].strip()
    return stripped


def matches_explicit_dutch_answer(generated: str, requested: str) -> bool:
    """Allow article insertion while rejecting lexical replacement of an authored answer."""
    generated_normalized = " ".join(generated.casefold().split())
    requested_normalized = " ".join(requested.casefold().split())
    if generated_normalized == requested_normalized:
        return True

    if requested_normalized.startswith(("de ", "het ")):
        return False
    return _strip_dutch_article(generated_normalized) == requested_normalized


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


class SourceConcept(StrictModel):
    """One learner-facing prompt with one or more accepted Dutch answers."""

    entry_id: str | None = None
    dutch_answers: tuple[str, ...] = Field(min_length=1)
    translation_hint: str | None = None
    topic: str | None = None
    lesson: str | None = None
    exam_level: str | None = None

    @field_validator("entry_id")
    @classmethod
    def normalize_entry_id(cls, value: str | None) -> str | None:
        """Reject blank explicit IDs used for stable concept identity."""
        if value is not None and not value.strip():
            raise ValueError("entry_id must be null or non-empty")
        return value

    @field_validator("dutch_answers")
    @classmethod
    def validate_dutch_answers(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """Reject blank or repeated accepted answers after normalizing whitespace and case."""
        if any(not answer.strip() for answer in value):
            raise ValueError("Dutch answers must not be empty")
        normalized = [" ".join(answer.casefold().split()) for answer in value]
        if len(normalized) != len(set(normalized)):
            raise ValueError("Dutch answers must be unique after normalization")
        return value

    @field_validator("translation_hint")
    @classmethod
    def normalize_translation_hint(cls, value: str | None) -> str | None:
        """Reject blank translation hints."""
        if value is not None and not value.strip():
            raise ValueError("translation_hint must be null or non-empty")
        return value

    @model_validator(mode="after")
    def require_grouped_translation_hint(self) -> "SourceConcept":
        """Require an application-owned Russian front for grouped concepts."""
        if len(self.dutch_answers) > 1 and self.translation_hint is None:
            raise ValueError("grouped concepts must include a Russian translation hint")
        return self

    def source_text(self) -> str:
        """Return the Dutch side of the authoring line in its original order."""
        return " | ".join(self.dutch_answers)

    def identity_key(self) -> str:
        """Return the stable concept identity used for duplicate detection and note GUIDs."""
        if self.entry_id is not None:
            return f"explicit:{' '.join(self.entry_id.casefold().split())}"

        first_answer = " ".join(self.dutch_answers[0].casefold().split())
        if self.translation_hint is None:
            return f"implicit:{first_answer}"
        hint = " ".join(self.translation_hint.casefold().split())
        return f"implicit:{first_answer}|hint:{hint}"

    def source_items(self) -> tuple["SourceItem", ...]:
        """Expand the concept into independently generated and cached Dutch answers."""
        is_grouped = len(self.dutch_answers) > 1
        return tuple(
            SourceItem(
                entry_id=self.entry_id if answer_index == 0 else None,
                text=answer,
                translation_hint=self.translation_hint,
                topic=self.topic,
                lesson=self.lesson,
                exam_level=self.exam_level,
                concept=self if is_grouped else None,
                answer_index=answer_index,
            )
            for answer_index, answer in enumerate(self.dutch_answers)
        )


class SourceItem(StrictModel):
    """Input item plus run-time metadata."""

    entry_id: str | None = None
    text: str
    translation_hint: str | None = None
    topic: str | None = None
    lesson: str | None = None
    exam_level: str | None = None
    concept: SourceConcept | None = Field(default=None, exclude=True, repr=False)
    answer_index: int = Field(default=0, ge=0, exclude=True)

    @field_validator("entry_id")
    @classmethod
    def normalize_entry_id(cls, value: str | None) -> str | None:
        """Reject blank explicit IDs used for stable cache and Anki identity."""
        if value is not None and not value.strip():
            raise ValueError("entry_id must be null or non-empty")
        return value

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

    def identity_key(self) -> str:
        """Return the stable logical identity used for duplicate detection."""
        if self.entry_id is not None:
            return f"explicit:{' '.join(self.entry_id.casefold().split())}"

        word = " ".join(self.text.casefold().split())
        if self.translation_hint is None:
            return f"implicit:{word}"
        hint = " ".join(self.translation_hint.casefold().split())
        return f"implicit:{word}|hint:{hint}"

    @model_validator(mode="after")
    def validate_concept_membership(self) -> "SourceItem":
        """Keep runtime-only concept metadata aligned with this answer leaf."""
        if self.concept is None:
            if self.answer_index != 0:
                raise ValueError("standalone source items must use answer_index 0")
            return self

        if len(self.concept.dutch_answers) < 2:
            raise ValueError("only grouped source items may reference a concept")
        if self.answer_index >= len(self.concept.dutch_answers):
            raise ValueError("answer_index is outside the concept's Dutch answers")
        expected_answer = self.concept.dutch_answers[self.answer_index]
        if " ".join(self.text.casefold().split()) != " ".join(expected_answer.casefold().split()):
            raise ValueError("source item text does not match its concept answer_index")
        if self.translation_hint != self.concept.translation_hint:
            raise ValueError("source item translation_hint must match its concept")
        if (self.topic, self.lesson, self.exam_level) != (
            self.concept.topic,
            self.concept.lesson,
            self.concept.exam_level,
        ):
            raise ValueError("source item learning context must match its concept")
        expected_entry_id = self.concept.entry_id if self.answer_index == 0 else None
        if self.entry_id != expected_entry_id:
            raise ValueError("only the first grouped answer may inherit the concept entry_id")
        return self

    def concept_identity_key(self) -> str:
        """Return the learner-facing concept identity for this answer leaf."""
        if self.concept is not None:
            return self.concept.identity_key()
        return self.identity_key()

    @property
    def accepted_dutch_answers(self) -> tuple[str, ...]:
        """Return every explicitly authored answer accepted for this prompt."""
        if self.concept is not None:
            return self.concept.dutch_answers
        return (self.text,)


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
    noun_number: NounNumber | None = None
    front_hint: str | None = Field(
        default=None,
        description=(
            "Russian-only noun meaning or sense hint. Do not include Dutch text, "
            "plural forms, or plural-recall wording; the application adds the "
            "plural-recall question for countable nouns."
        ),
    )
    verb_forms: VerbForms | None = None
    adjective_forms: AdjectiveForms | None = None

    @field_validator("part_of_speech", mode="before")
    @classmethod
    def normalize_part_of_speech(cls, value: object) -> object:
        """Accept part-of-speech values regardless of letter case."""
        return _normalize_enum_value(value, PartOfSpeech)

    @field_validator("noun_number", mode="before")
    @classmethod
    def normalize_noun_number(cls, value: object) -> object:
        """Accept noun-number values regardless of letter case."""
        return _normalize_enum_value(value, NounNumber)

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
            if self.noun_number is None:
                self.noun_number = (
                    NounNumber.COUNTABLE
                    if self.plural_form is not None
                    else NounNumber.UNCOUNTABLE
                )
            if not _starts_with_dutch_article(self.dutch_word):
                raise ValueError("nouns must include article 'de' or 'het' in dutch_word")
            if not self.front_hint:
                raise ValueError("nouns must include front_hint")
            if self.noun_number != NounNumber.COUNTABLE and "множественное число" in self.front_hint:
                raise ValueError("nouns without a plural drill must not prompt plural recall")
            if self.verb_forms is not None:
                raise ValueError("nouns must not include verb_forms")
            if self.adjective_forms is not None:
                raise ValueError("nouns must not include adjective_forms")
            if self.noun_number == NounNumber.COUNTABLE:
                if self.plural_form is None:
                    raise ValueError("countable nouns must include plural_form")
                if _starts_with_dutch_article(self.plural_form):
                    raise ValueError("countable noun plural_form must not include article")
                if _strip_dutch_article(self.dutch_word).casefold() == self.plural_form.casefold():
                    raise ValueError(
                        "countable noun dutch_word must be singular; "
                        "plural_form must differ from article-stripped dutch_word"
                    )
                self._require_exact_example_kinds(
                    {FormExampleKind.SINGULAR, FormExampleKind.PLURAL},
                    "countable nouns",
                )
            else:
                if self.plural_form is not None:
                    raise ValueError(f"{self.noun_number.value} nouns must not include plural_form")
                self._normalize_single_noun_form_example_kind()
                self._require_exact_example_kinds(
                    {FormExampleKind.DEFAULT},
                    f"{self.noun_number.value} nouns",
                )
            return self

        if self.part_of_speech == PartOfSpeech.VERB:
            if self.verb_forms is None:
                raise ValueError("verbs must include verb_forms")
            if self.plural_form or self.front_hint:
                raise ValueError("verbs must not include noun-only fields")
            if self.adjective_forms is not None:
                raise ValueError("verbs must not include adjective_forms")
            if self.noun_number is not None:
                raise ValueError("non-nouns must set noun_number to null")
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
            if self.noun_number is not None:
                raise ValueError("non-nouns must set noun_number to null")
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
        if self.noun_number is not None:
            raise ValueError("non-nouns must set noun_number to null")
        self._require_visible_form_examples()
        self._require_exact_example_kinds({FormExampleKind.DEFAULT}, "single-form words")
        return self

    def _require_visible_form_examples(self) -> None:
        """Require selected forms to be visible for non-verb examples."""
        for example in self.form_examples:
            form_tokens = set(_word_tokens(example.form))
            example_tokens = set(_word_tokens(example.example_sentence_nl))
            if form_tokens and not form_tokens.issubset(example_tokens):
                raise ValueError(
                    f"form {example.form!r} must appear in "
                    f"example_sentence_nl {example.example_sentence_nl!r}"
                )

    def _normalize_single_noun_form_example_kind(self) -> None:
        """Treat one number-labelled example as the default for non-countable drills."""
        if len(self.form_examples) != 1:
            return
        example = self.form_examples[0]
        if example.kind in {FormExampleKind.SINGULAR, FormExampleKind.PLURAL}:
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
