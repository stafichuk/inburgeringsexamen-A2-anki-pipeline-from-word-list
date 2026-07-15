"""Local file-based caching for generated cards."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .models import GeneratedCard, SourceItem

CACHE_FORMAT_VERSION = 2


def normalize_word(value: str) -> str:
    """Normalize source words for stable cache keys."""
    return " ".join(value.strip().lower().split())


def normalize_text(value: str) -> str:
    """Normalize free-form text for stable cache keys without changing case."""
    return " ".join(value.strip().split())


class CardCache:
    """File-based cache for generated card payloads."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)

    def build_key(
        self,
        source_item: SourceItem,
    ) -> str:
        """Create a stable cache key for a source item and generation context."""
        payload = {
            "cache_format_version": CACHE_FORMAT_VERSION,
            "identity": source_item.identity_key(),
            "word": normalize_word(source_item.text),
            "translation_hint": normalize_text(source_item.translation_hint or ""),
            "topic": source_item.topic or "",
            "lesson": source_item.lesson or "",
            "exam_level": source_item.exam_level or "",
        }
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _path_for_key(self, key: str) -> Path:
        """Return the cache file path for a given key."""
        return self.directory / f"{key}.json"

    def _refresh_manifest_paths(self) -> list[Path]:
        """Return active force-refresh manifests."""
        return sorted(self.directory.glob(".refresh-*.json"))

    def _load_refresh_keys(self, path: Path) -> set[str]:
        """Load one force-refresh manifest, failing closed if it is corrupt."""
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"invalid cache refresh manifest: {path}") from exc

        pending_keys = payload.get("pending_keys") if isinstance(payload, dict) else None
        if not isinstance(pending_keys, list) or not all(
            isinstance(key, str) for key in pending_keys
        ):
            raise ValueError(f"invalid cache refresh manifest: {path}")
        return set(pending_keys)

    def _write_refresh_keys(self, path: Path, pending_keys: set[str]) -> None:
        """Atomically persist the unresolved keys for one force refresh."""
        temporary_path = path.with_name(f".{path.name}.tmp")
        payload = {
            "cache_format_version": CACHE_FORMAT_VERSION,
            "pending_keys": sorted(pending_keys),
        }
        temporary_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary_path.replace(path)

    def begin_refresh(self, source_items: list[SourceItem]) -> None:
        """Mark a complete input set stale before a resumable forced refresh."""
        pending_keys = {self.build_key(source_item) for source_item in source_items}
        serialized = json.dumps(sorted(pending_keys), separators=(",", ":"))
        manifest_id = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        self._write_refresh_keys(
            self.directory / f".refresh-{manifest_id}.json",
            pending_keys,
        )

    def _is_refresh_pending(self, key: str) -> bool:
        """Return whether an unfinished forced refresh suppresses this cache key."""
        return any(key in self._load_refresh_keys(path) for path in self._refresh_manifest_paths())

    def _complete_refresh_key(self, key: str) -> None:
        """Mark a newly persisted card complete in every active refresh manifest."""
        for path in self._refresh_manifest_paths():
            pending_keys = self._load_refresh_keys(path)
            if key not in pending_keys:
                continue
            pending_keys.remove(key)
            if pending_keys:
                self._write_refresh_keys(path, pending_keys)
            else:
                path.unlink(missing_ok=True)

    def get(
        self,
        source_item: SourceItem,
    ) -> GeneratedCard | None:
        """Return a cached card if present and valid."""
        key = self.build_key(source_item)
        if self._is_refresh_pending(key):
            return None
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
        key = self.build_key(source_item)
        path = self._path_for_key(key)
        payload = {
            "cache_format_version": CACHE_FORMAT_VERSION,
            "source_entry_id": source_item.entry_id,
            "source_word": source_item.text,
            "source_translation_hint": source_item.translation_hint,
            "topic": source_item.topic,
            "lesson": source_item.lesson,
            "exam_level": source_item.exam_level,
            "model_name": model_name,
            "prompt_version": prompt_version,
            "card": card.model_dump(mode="json"),
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        self._complete_refresh_key(key)

    def delete(self, source_item: SourceItem) -> None:
        """Delete the current-format cache entry for a source item if present."""
        self._path_for_key(self.build_key(source_item)).unlink(missing_ok=True)
