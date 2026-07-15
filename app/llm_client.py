"""OpenAI-compatible LLM client with coordinated batch generation."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any
from urllib import error, request

from pydantic import ValidationError

from .config import LLMSettings
from .models import GeneratedCard, SourceItem
from .prompts import build_messages

LOGGER = logging.getLogger(__name__)


class LLMClientError(RuntimeError):
    """Base error for LLM generation failures."""


class LLMResponseFormatError(LLMClientError):
    """Raised when the provider returns malformed or invalid content."""

    def __init__(self, message: str, *, raw_text: str | None = None) -> None:
        super().__init__(message)
        self.raw_text = raw_text

    def __str__(self) -> str:
        message = super().__str__()
        if self.raw_text is None:
            return message
        raw_text = self.raw_text.strip() or "<empty response>"
        return f"{message}\nLLM response:\n{raw_text}"


@dataclass(slots=True)
class BatchParseResult:
    """Valid cards and per-source validation errors from one response."""

    cards: dict[int, GeneratedCard] = field(default_factory=dict)
    errors: dict[int, str] = field(default_factory=dict)
    global_errors: list[str] = field(default_factory=list)

    def validation_feedback(self) -> str:
        """Return compact feedback suitable for a retry prompt."""
        lines = [f"source_id {source_id}: {reason}" for source_id, reason in sorted(self.errors.items())]
        lines.extend(self.global_errors)
        return "\n".join(lines) or "The response did not resolve every requested source_id."


@dataclass(slots=True)
class BatchGenerationResult:
    """Cards accepted across all attempts plus unresolved source IDs."""

    cards: dict[int, GeneratedCard] = field(default_factory=dict)
    failures: dict[int, str] = field(default_factory=dict)


def extract_json_object(raw_text: str) -> str:
    """Attempt to extract a JSON object from raw model output."""
    stripped = raw_text.strip()
    if stripped.startswith("```"):
        lines = [line for line in stripped.splitlines() if not line.strip().startswith("```")]
        stripped = "\n".join(lines).strip()

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise LLMResponseFormatError("no JSON object found in model response")
    return stripped[start : end + 1]


def _decode_json_object(raw_text: str, *, allow_json_repair: bool) -> dict[str, Any]:
    """Decode one JSON object and retain the raw response on errors."""
    try:
        json_text = extract_json_object(raw_text) if allow_json_repair else raw_text.strip()
    except LLMResponseFormatError as exc:
        raise LLMResponseFormatError(str(exc), raw_text=raw_text) from exc

    try:
        payload = json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise LLMResponseFormatError(f"invalid JSON payload: {exc}", raw_text=raw_text) from exc
    if not isinstance(payload, dict):
        raise LLMResponseFormatError("response must be one JSON object", raw_text=raw_text)
    return payload


def parse_generated_card(raw_text: str, *, allow_json_repair: bool = True) -> GeneratedCard:
    """Parse a single GeneratedCard for backwards-compatible callers."""
    payload = _decode_json_object(raw_text, allow_json_repair=allow_json_repair)
    try:
        return GeneratedCard.model_validate(payload)
    except ValidationError as exc:
        raise LLMResponseFormatError(f"JSON does not match schema: {exc}", raw_text=raw_text) from exc


def parse_generated_cards(
    raw_text: str,
    *,
    expected_items: Mapping[int, SourceItem],
    allow_json_repair: bool = True,
) -> BatchParseResult:
    """Validate cards and their source association independently."""
    payload = _decode_json_object(raw_text, allow_json_repair=allow_json_repair)
    raw_cards = payload.get("cards")
    if not isinstance(raw_cards, list):
        raise LLMResponseFormatError("response field 'cards' must be a JSON array", raw_text=raw_text)

    result = BatchParseResult()
    unexpected_fields = sorted(set(payload) - {"cards"})
    if unexpected_fields:
        result.global_errors.append(f"Ignored unexpected top-level fields: {', '.join(unexpected_fields)}")

    entries_by_id: dict[int, list[dict[str, Any]]] = {}
    for position, raw_item in enumerate(raw_cards, start=1):
        if not isinstance(raw_item, dict):
            result.global_errors.append(f"cards[{position}] must be an object")
            continue
        source_id = raw_item.get("source_id")
        if isinstance(source_id, bool) or not isinstance(source_id, int):
            result.global_errors.append(f"cards[{position}].source_id must be an integer")
            continue
        if source_id not in expected_items:
            result.global_errors.append(f"Ignored unknown source_id {source_id}")
            continue
        entries_by_id.setdefault(source_id, []).append(raw_item)

    for source_id in sorted(expected_items):
        entries = entries_by_id.get(source_id, [])
        if not entries:
            result.errors[source_id] = "missing from response"
            continue
        if len(entries) > 1:
            result.errors[source_id] = "returned more than once"
            continue

        raw_item = entries[0]
        echoed_input = raw_item.get("input_item")
        expected_input = expected_items[source_id].text
        if echoed_input != expected_input:
            result.errors[source_id] = (
                "input_item does not exactly match the request: "
                f"expected {expected_input!r}, got {echoed_input!r}"
            )
            continue

        if "translation_hint" not in raw_item:
            result.errors[source_id] = "translation_hint is missing from the response wrapper"
            continue
        echoed_hint = raw_item["translation_hint"]
        expected_hint = expected_items[source_id].translation_hint
        if echoed_hint != expected_hint:
            result.errors[source_id] = (
                "translation_hint does not exactly match the request: "
                f"expected {expected_hint!r}, got {echoed_hint!r}"
            )
            continue

        raw_card = raw_item.get("card")
        try:
            result.cards[source_id] = GeneratedCard.model_validate(raw_card)
        except ValidationError as exc:
            result.errors[source_id] = f"card does not match schema: {exc}"

    return result


@dataclass(slots=True)
class LLMClient:
    """Chat-completions client for resumable coordinated batch generation."""

    settings: LLMSettings
    json_repair_enabled: bool = True

    def generate_cards(
        self,
        pending_items: list[tuple[int, SourceItem]],
        existing_cards: list[tuple[SourceItem, GeneratedCard]],
        *,
        on_card_accepted: Callable[[int, GeneratedCard], None] | None = None,
    ) -> BatchGenerationResult:
        """Generate pending cards, retrying only unresolved source IDs."""
        requested_by_id = dict(pending_items)
        if len(requested_by_id) != len(pending_items):
            raise ValueError("pending source IDs must be unique")
        if not requested_by_id:
            return BatchGenerationResult()

        pending = dict(requested_by_id)
        accepted: dict[int, GeneratedCard] = {}
        context = list(existing_cards)
        failures = {source_id: "not generated" for source_id in pending}
        validation_error: str | None = None
        attempts = self.settings.max_retries + 1

        for attempt in range(1, attempts + 1):
            try:
                response_text = self._request_completion(
                    list(pending.items()),
                    context,
                    validation_error=validation_error,
                )
                parsed = parse_generated_cards(
                    response_text,
                    expected_items=pending,
                    allow_json_repair=self.json_repair_enabled,
                )
                for source_id, card in parsed.cards.items():
                    accepted[source_id] = card
                    context.append((requested_by_id[source_id], card))
                    pending.pop(source_id, None)
                    failures.pop(source_id, None)
                    if on_card_accepted is not None:
                        on_card_accepted(source_id, card)

                for source_id in pending:
                    failures[source_id] = parsed.errors.get(source_id, "not resolved by response")
                validation_error = parsed.validation_feedback()
                for message in parsed.global_errors:
                    LOGGER.warning("LLM batch response: %s", message)
            except (LLMClientError, error.URLError, TimeoutError) as exc:
                validation_error = str(exc)
                for source_id in pending:
                    failures[source_id] = str(exc)

            if not pending:
                break
            if attempt >= attempts:
                break

            sleep_seconds = self.settings.retry_backoff_seconds * attempt
            LOGGER.warning(
                "LLM batch attempt %s/%s left %s card(s) unresolved. Retrying in %.1fs.",
                attempt,
                attempts,
                len(pending),
                sleep_seconds,
            )
            time.sleep(sleep_seconds)

        return BatchGenerationResult(cards=accepted, failures=failures)

    def generate_card(self, source_item: SourceItem) -> GeneratedCard:
        """Generate one card through the batch interface."""
        result = self.generate_cards([(1, source_item)], [])
        if 1 in result.cards:
            return result.cards[1]
        reason = result.failures.get(1, "unknown generation failure")
        raise LLMClientError(f"failed to generate card for '{source_item.text}': {reason}")

    def _request_completion(
        self,
        pending_items: list[tuple[int, SourceItem]],
        existing_cards: list[tuple[SourceItem, GeneratedCard]],
        *,
        validation_error: str | None = None,
    ) -> str:
        """Send one chat-completions request and return the raw content string."""
        payload = {
            "model": self.settings.model_name,
            "messages": build_messages(
                pending_items,
                existing_cards,
                validation_error=validation_error,
            ),
            "temperature": self.settings.temperature,
            "max_tokens": self.settings.max_tokens,
        }

        body = json.dumps(payload).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.settings.api_token}",
            "Content-Type": "application/json",
        }
        headers.update(self.settings.custom_headers)

        http_request = request.Request(
            self.settings.base_url,
            data=body,
            headers=headers,
            method="POST",
        )

        try:
            with request.urlopen(http_request, timeout=self.settings.timeout_seconds) as response:
                raw_response = response.read().decode("utf-8")
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise LLMClientError(f"provider returned HTTP {exc.code}: {detail}") from exc

        return self._extract_message_content(raw_response)

    def _extract_message_content(self, raw_response: str) -> str:
        """Extract the first chat-completions message content from provider JSON."""
        try:
            payload = json.loads(raw_response)
        except json.JSONDecodeError as exc:
            raise LLMClientError(f"provider returned invalid JSON: {exc}") from exc

        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise LLMClientError("provider response missing choices")

        message = choices[0].get("message", {})
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts = [
                part["text"]
                for part in content
                if isinstance(part, dict)
                and part.get("type") == "text"
                and isinstance(part.get("text"), str)
            ]
            if text_parts:
                return "\n".join(text_parts)
        raise LLMClientError("provider response missing text message content")
