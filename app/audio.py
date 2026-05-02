"""Audio generation backends."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Protocol
from urllib import error, request
from xml.sax.saxutils import escape as escape_xml

from .config import AudioSettings, AzureAudioSettings


AZURE_MP3_OUTPUT_FORMAT = "audio-24khz-96kbitrate-mono-mp3"


class AudioGenerationError(RuntimeError):
    """Raised when an audio backend cannot synthesize speech."""


class AudioGenerator(Protocol):
    """Protocol for swappable text-to-speech backends."""

    def generate_audio(self, text: str, *, label: str) -> Path:
        """Generate or return a cached audio file for the given text."""


@dataclass(slots=True)
class AzureTextToSpeechGenerator:
    """Azure Speech REST implementation of the audio generator."""

    settings: AzureAudioSettings
    directory: Path
    timeout_seconds: float = 60.0
    output_format: str = AZURE_MP3_OUTPUT_FORMAT

    def __post_init__(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)

    def generate_audio(self, text: str, *, label: str) -> Path:
        """Generate or return a cached MP3 file for text."""
        if not text.strip():
            raise AudioGenerationError("cannot synthesize blank text")

        output_path = self._path_for_text(text, label=label)
        if output_path.exists() and output_path.stat().st_size > 0:
            return output_path

        audio_bytes = self._request_audio(text)
        if not audio_bytes:
            raise AudioGenerationError("Azure TTS returned an empty audio payload")

        temporary_path = output_path.with_name(f".{output_path.name}.{time.monotonic_ns()}.tmp")
        temporary_path.write_bytes(audio_bytes)
        temporary_path.replace(output_path)
        return output_path

    def _path_for_text(self, text: str, *, label: str) -> Path:
        """Create a stable cache path for synthesized audio."""
        payload = {
            "provider": "azure",
            "voice": self.settings.voice,
            "format": self.output_format,
            "label": label,
            "text": text,
        }
        serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:20]
        safe_label = "".join(character if character.isascii() and character.isalnum() else "-" for character in label)
        safe_label = "-".join(part for part in safe_label.lower().split("-") if part) or "audio"
        return self.directory / f"{safe_label}-{digest}.mp3"

    def _request_audio(self, text: str) -> bytes:
        """Call Azure Speech REST and return MP3 bytes."""
        api_key = self.settings.api_key
        voice = self.settings.voice
        if api_key is None or voice is None:
            raise AudioGenerationError("Azure TTS requires api_key and voice")

        http_request = request.Request(
            self._resolve_endpoint(),
            data=self._build_ssml(text).encode("utf-8"),
            headers={
                "Content-Type": "application/ssml+xml",
                "Ocp-Apim-Subscription-Key": api_key,
                "X-Microsoft-OutputFormat": self.output_format,
                "User-Agent": "dutch-a2-anki-pipeline",
            },
            method="POST",
        )

        try:
            with request.urlopen(http_request, timeout=self.timeout_seconds) as response:
                return response.read()
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise AudioGenerationError(f"Azure TTS returned HTTP {exc.code}: {detail}") from exc
        except (error.URLError, TimeoutError) as exc:
            raise AudioGenerationError(f"Azure TTS request failed: {exc}") from exc

    def _resolve_endpoint(self) -> str:
        """Resolve either an explicit endpoint or a region into the Azure synthesis URL."""
        if self.settings.endpoint is not None:
            return self.settings.endpoint.rstrip("/")
        if self.settings.region is None:
            raise AudioGenerationError("Azure TTS requires region or endpoint")
        return f"https://{self.settings.region}.tts.speech.microsoft.com/cognitiveservices/v1"

    def _build_ssml(self, text: str) -> str:
        """Build Azure-compatible SSML for the configured voice."""
        voice = self.settings.voice
        if voice is None:
            raise AudioGenerationError("Azure TTS requires voice")

        language = _language_from_voice(voice)
        return (
            f'<speak version="1.0" xml:lang="{language}">'
            f'<voice xml:lang="{language}" name="{escape_xml(voice)}">'
            f"{escape_xml(text)}"
            "</voice>"
            "</speak>"
        )


def build_audio_generator(settings: AudioSettings) -> AudioGenerator | None:
    """Create the configured audio generator, if audio is enabled."""
    if not settings.enabled:
        return None
    if settings.provider == "azure":
        return AzureTextToSpeechGenerator(settings.azure, directory=settings.directory)
    raise AudioGenerationError(f"unsupported audio provider: {settings.provider}")


def _language_from_voice(voice: str) -> str:
    """Infer an Azure BCP-47 language tag from a voice name."""
    parts = voice.split("-")
    if len(parts) >= 2 and len(parts[0]) == 2 and len(parts[1]) == 2:
        return f"{parts[0].lower()}-{parts[1].upper()}"
    return "nl-NL"
