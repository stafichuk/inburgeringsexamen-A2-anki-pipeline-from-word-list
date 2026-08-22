"""Configuration loading and CLI override resolution."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field, ValidationError, field_validator, model_validator

from .models import StrictModel

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.11+ target
    tomllib = None  # type: ignore[assignment]


class LLMSettings(StrictModel):
    """Settings for an OpenAI-compatible chat-completions endpoint."""

    base_url: str
    api_token: str
    model_name: str
    custom_headers: dict[str, str] = Field(default_factory=dict)
    timeout_seconds: float = 60.0
    max_retries: int = 2
    retry_backoff_seconds: float = 1.0
    temperature: float = 0.2
    max_tokens: int = 65536

    @field_validator("base_url", "api_token", "model_name")
    @classmethod
    def ensure_required_text(cls, value: str) -> str:
        """Reject blank required settings."""
        if not value.strip():
            raise ValueError("LLM settings must not be blank")
        return value


class DeckSettings(StrictModel):
    """Output deck settings."""

    deck_name: str | None = None
    model_name: str = "Dutch A2 Inburgering Vocabulary"
    deck_id_seed: str = "dutch-a2-inburgering"


class GenerationSettings(StrictModel):
    """Generation defaults and validation behavior."""

    default_topic: str | None = None
    default_lesson: str | None = None
    default_exam_level: str = "A2 Inburgering Spreken"
    json_repair_enabled: bool = True


class CacheSettings(StrictModel):
    """Local cache settings."""

    directory: Path = Path(".cache/cards")


class AzureAudioSettings(StrictModel):
    """Azure Text to Speech settings."""

    region: str | None = None
    endpoint: str | None = None
    api_key: str | None = None
    voice: str | None = None

    @field_validator("region", "endpoint", "api_key", "voice")
    @classmethod
    def normalize_optional_text(cls, value: str | None) -> str | None:
        """Reject blank optional text values."""
        if value is not None and not value.strip():
            raise ValueError("audio settings must be null or non-empty")
        return value


class AudioSettings(StrictModel):
    """Optional audio generation settings."""

    enabled: bool = False
    provider: Literal["azure"] = "azure"
    directory: Path = Path(".cache/audio")
    azure: AzureAudioSettings = Field(default_factory=AzureAudioSettings)


class LoggingSettings(StrictModel):
    """Logging defaults."""

    level: str = "INFO"

    @field_validator("level")
    @classmethod
    def normalize_level(cls, value: str) -> str:
        """Normalize log level names."""
        return value.upper()


class AppSettings(StrictModel):
    """Top-level application settings."""

    llm: LLMSettings | None = None
    deck: DeckSettings = Field(default_factory=DeckSettings)
    generation: GenerationSettings = Field(default_factory=GenerationSettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    audio: AudioSettings = Field(default_factory=AudioSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)

    @model_validator(mode="after")
    def validate_audio_settings(self) -> "AppSettings":
        """Require complete provider settings when audio generation is enabled."""
        if not self.audio.enabled:
            return self

        azure = self.audio.azure
        missing: list[str] = []
        if azure.api_key is None:
            missing.append("audio.azure.api_key")
        if azure.voice is None:
            missing.append("audio.azure.voice")
        if azure.region is None and azure.endpoint is None:
            missing.append("audio.azure.region or audio.azure.endpoint")
        if missing:
            raise ValueError(f"missing required audio settings: {', '.join(missing)}")
        return self


def _read_config_file(path: Path) -> dict[str, Any]:
    """Load a config file as a dictionary based on its extension."""
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")

    suffix = path.suffix.lower()
    raw_text = path.read_text(encoding="utf-8")
    if suffix in {".yaml", ".yml"}:
        data = yaml.safe_load(raw_text) or {}
    elif suffix == ".json":
        data = json.loads(raw_text)
    elif suffix == ".toml":
        if tomllib is None:
            raise RuntimeError("TOML config requires Python 3.11+ or tomllib support")
        data = tomllib.loads(raw_text)
    else:
        raise ValueError(f"unsupported config format: {suffix}")

    if not isinstance(data, dict):
        raise ValueError("config file must contain a top-level mapping")
    return data


def _merge_dicts(base: dict[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge override values into a copy of the base mapping."""
    merged = dict(base)
    for key, value in overrides.items():
        if value is None:
            continue
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, Mapping):
            merged[key] = _merge_dicts(existing, value)
        else:
            merged[key] = value
    return merged


def load_settings(
    config_path: Path | None = None,
    overrides: Mapping[str, Any] | None = None,
    *,
    require_llm: bool = True,
) -> AppSettings:
    """Load and validate settings from config file plus optional CLI overrides."""
    raw_config: dict[str, Any] = {}
    if config_path is not None:
        raw_config = _read_config_file(config_path)

    merged = _merge_dicts(raw_config, overrides or {})
    try:
        settings = AppSettings.model_validate(merged)
    except ValidationError as exc:
        raise ValueError(f"invalid configuration: {exc}") from exc
    if require_llm and settings.llm is None:
        raise ValueError("missing required LLM settings for word-list generation mode")
    return settings
