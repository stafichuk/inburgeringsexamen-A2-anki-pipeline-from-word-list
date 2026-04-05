"""Local file-based caching for generated cards."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .models import GeneratedCard, SourceItem


def normalize_word(value: str) -> str:
    """Normalize source words for stable cache keys."""
    return " ".join(value.strip().lower().split())


class CardCache:
    """File-based cache for generated card payloads."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)

    def build_key(
        self,
        source_item: SourceItem,
        *,
        model_name: str,
        prompt_version: str,
    ) -> str:
        """Create a stable cache key for a source item and generation context."""
        payload = {
            "word": normalize_word(source_item.text),
            "topic": source_item.topic or "",
            "lesson": source_item.lesson or "",
            "exam_level": source_item.exam_level or "",
            "model_name": model_name,
            "prompt_version": prompt_version,
        }
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _path_for_key(self, key: str) -> Path:
        """Return the cache file path for a given key."""
        return self.directory / f"{key}.json"

    def get(
        self,
        source_item: SourceItem,
        *,
        model_name: str,
        prompt_version: str,
    ) -> GeneratedCard | None:
        """Return a cached card if present and valid."""
        key = self.build_key(source_item, model_name=model_name, prompt_version=prompt_version)
        path = self._path_for_key(key)
        if not path.exists():
            return None

        try:
            raw_data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
            card_payload = raw_data.get("card", raw_data)
            return GeneratedCard.model_validate(card_payload)
        except (json.JSONDecodeError, ValidationError):
            return None

    def set(
        self,
        source_item: SourceItem,
        card: GeneratedCard,
        *,
        model_name: str,
        prompt_version: str,
    ) -> None:
        """Store a generated card in the cache."""
        key = self.build_key(source_item, model_name=model_name, prompt_version=prompt_version)
        path = self._path_for_key(key)
        payload = {
            "source_word": source_item.text,
            "topic": source_item.topic,
            "lesson": source_item.lesson,
            "exam_level": source_item.exam_level,
            "model_name": model_name,
            "prompt_version": prompt_version,
            "card": card.model_dump(mode="json"),
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
