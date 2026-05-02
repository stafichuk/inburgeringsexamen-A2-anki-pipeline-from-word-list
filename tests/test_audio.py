from __future__ import annotations

import io
from pathlib import Path
from urllib import error

import pytest

import app.audio as audio_module
from app.audio import AudioGenerationError, AzureTextToSpeechGenerator
from app.config import AzureAudioSettings


class FakeResponse:
    def __init__(self, body: bytes) -> None:
        self.body = body

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args) -> None:  # type: ignore[no-untyped-def]
        return None

    def read(self) -> bytes:
        return self.body


def make_generator(tmp_path: Path, *, endpoint: str | None = None) -> AzureTextToSpeechGenerator:
    return AzureTextToSpeechGenerator(
        AzureAudioSettings(
            region="westeurope",
            endpoint=endpoint,
            api_key="secret",
            voice="nl-NL-FennaNeural",
        ),
        directory=tmp_path,
    )


def test_azure_tts_writes_audio_and_sends_expected_request(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    generator = make_generator(tmp_path)
    captured = {}

    def fake_urlopen(http_request, timeout):  # type: ignore[no-untyped-def]
        captured["url"] = http_request.full_url
        captured["timeout"] = timeout
        captured["body"] = http_request.data.decode("utf-8")
        captured["api_key"] = http_request.get_header("Ocp-apim-subscription-key")
        captured["format"] = http_request.get_header("X-microsoft-outputformat")
        return FakeResponse(b"mp3-bytes")

    monkeypatch.setattr(audio_module.request, "urlopen", fake_urlopen)

    output_path = generator.generate_audio("school & les", label="word")

    assert output_path.exists()
    assert output_path.read_bytes() == b"mp3-bytes"
    assert captured["url"] == "https://westeurope.tts.speech.microsoft.com/cognitiveservices/v1"
    assert captured["timeout"] == 60.0
    assert captured["api_key"] == "secret"
    assert captured["format"] == audio_module.AZURE_MP3_OUTPUT_FORMAT
    assert 'xml:lang="nl-NL"' in captured["body"]
    assert 'name="nl-NL-FennaNeural"' in captured["body"]
    assert "school &amp; les" in captured["body"]


def test_azure_tts_reuses_existing_audio_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    generator = make_generator(tmp_path)
    output_path = generator._path_for_text("leren", label="word")
    output_path.write_bytes(b"existing")

    def fake_urlopen(http_request, timeout):  # type: ignore[no-untyped-def]
        raise AssertionError("Azure should not be called for an existing audio file")

    monkeypatch.setattr(audio_module.request, "urlopen", fake_urlopen)

    assert generator.generate_audio("leren", label="word") == output_path
    assert output_path.read_bytes() == b"existing"


def test_azure_endpoint_wins_over_region(tmp_path: Path) -> None:
    generator = make_generator(tmp_path, endpoint="https://custom.example.test/tts")

    assert generator._resolve_endpoint() == "https://custom.example.test/tts"


def test_azure_tts_reports_http_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    generator = make_generator(tmp_path)

    def fake_urlopen(http_request, timeout):  # type: ignore[no-untyped-def]
        raise error.HTTPError(
            url=http_request.full_url,
            code=401,
            msg="Unauthorized",
            hdrs={},
            fp=io.BytesIO(b"bad key"),
        )

    monkeypatch.setattr(audio_module.request, "urlopen", fake_urlopen)

    with pytest.raises(AudioGenerationError, match="HTTP 401"):
        generator.generate_audio("leren", label="word")
