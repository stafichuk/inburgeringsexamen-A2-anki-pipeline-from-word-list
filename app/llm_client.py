"""OpenAI-compatible LLM client with strict JSON parsing and retries."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
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


def parse_generated_card(raw_text: str, *, allow_json_repair: bool = True) -> GeneratedCard:
    """Parse and validate a GeneratedCard from raw model output."""
    try:
        json_text = extract_json_object(raw_text) if allow_json_repair else raw_text.strip()
    except LLMResponseFormatError as exc:
        raise LLMResponseFormatError(str(exc), raw_text=raw_text) from exc

    try:
        payload = json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise LLMResponseFormatError(f"invalid JSON payload: {exc}", raw_text=raw_text) from exc

    try:
        return GeneratedCard.model_validate(payload)
    except ValidationError as exc:
        raise LLMResponseFormatError(f"JSON does not match schema: {exc}", raw_text=raw_text) from exc


@dataclass(slots=True)
class LLMClient:
    """Chat-completions client for generating validated card payloads."""

    settings: LLMSettings
    json_repair_enabled: bool = True

    def generate_card(self, source_item: SourceItem) -> GeneratedCard:
        """Generate one validated card, retrying on transport and format errors."""
        attempts = self.settings.max_retries + 1
        last_error: Exception | None = None
`       previous_response: str | None = None
        validation_error: str | None = None

        for attempt in range(1, attempts + 1):
            try:
                response_text = self._request_completion(
                    source_item,
                    previous_response=previous_response,
                    validation_error=validation_error,
                )
                return parse_generated_card(response_text, allow_json_repair=self.json_repair_enabled)
            except (LLMClientError, error.URLError, TimeoutError) as exc:
                last_error = exc
                if isinstance(exc, LLMResponseFormatError):
                    previous_response = exc.raw_text
                    validation_error = str(exc.args[0]) if exc.args else str(exc)
                if attempt >= attempts:
                    break
                sleep_seconds = self.settings.retry_backoff_seconds * attempt
                LOGGER.warning(
                    "LLM generation failed for '%s' on attempt %s/%s: %s. Retrying in %.1fs.",
                    source_item.text,
                    attempt,
                    attempts,
                    exc,
                    sleep_seconds,
                )
                time.sleep(sleep_seconds)

        message = f"failed to generate card for '{source_item.text}'"
        if last_error is not None:
            raise LLMClientError(f"{message}: {last_error}") from last_error
        raise LLMClientError(message)

    def _request_completion(
        self,
        source_item: SourceItem,
        *,
        previous_response: str | None = None,
        validation_error: str | None = None,
    ) -> str:
        """Send a chat-completions request and return the raw content string."""
        payload = {
            "model": self.settings.model_name,
            "messages": build_messages(
                source_item,
                previous_response=previous_response,
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
            text_parts: list[str] = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str):
                    text_parts.append(part["text"])
            if text_parts:
                return "\n".join(text_parts)

        raise LLMClientError("provider response missing text message content")
