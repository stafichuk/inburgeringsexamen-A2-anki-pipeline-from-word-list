from pathlib import Path

import pytest
from pydantic import ValidationError

from app.config import AppSettings


def base_settings_payload(cache_dir: Path) -> dict:
    return {
        "llm": {
            "base_url": "https://example.invalid/v1/chat/completions",
            "api_token": "token",
            "model_name": "test-model",
        },
        "cache": {"directory": str(cache_dir / "cards")},
    }


def test_audio_settings_are_disabled_by_default(tmp_path: Path) -> None:
    settings = AppSettings.model_validate(base_settings_payload(tmp_path))

    assert settings.audio.enabled is False
    assert settings.audio.provider == "azure"
    assert settings.audio.directory == Path(".cache/audio")


def test_enabled_audio_requires_azure_credentials_and_location(tmp_path: Path) -> None:
    payload = base_settings_payload(tmp_path)
    payload["audio"] = {"enabled": True}

    with pytest.raises(ValidationError) as exc_info:
        AppSettings.model_validate(payload)

    message = str(exc_info.value)
    assert "audio.azure.api_key" in message
    assert "audio.azure.voice" in message
    assert "audio.azure.region or audio.azure.endpoint" in message


def test_enabled_audio_accepts_region_or_endpoint(tmp_path: Path) -> None:
    region_payload = base_settings_payload(tmp_path)
    region_payload["audio"] = {
        "enabled": True,
        "azure": {
            "region": "westeurope",
            "api_key": "key",
            "voice": "nl-NL-FennaNeural",
        },
    }
    endpoint_payload = base_settings_payload(tmp_path)
    endpoint_payload["audio"] = {
        "enabled": True,
        "azure": {
            "endpoint": "https://speech.example.test/cognitiveservices/v1",
            "api_key": "key",
            "voice": "nl-NL-FennaNeural",
        },
    }

    assert AppSettings.model_validate(region_payload).audio.azure.region == "westeurope"
    assert AppSettings.model_validate(endpoint_payload).audio.azure.endpoint == (
        "https://speech.example.test/cognitiveservices/v1"
    )
