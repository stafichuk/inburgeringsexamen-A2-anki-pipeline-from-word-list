"""Command-line interface for deck generation."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import logging
from pathlib import Path
from typing import Any

from .config import AppSettings, load_settings
from .pipeline import DeckGenerationPipeline


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="generate-deck",
        description="Generate an Anki .apkg deck for Dutch A2 vocabulary study.",
    )
    parser.add_argument("--input", required=True, help="Path to the input word list.")
    parser.add_argument("--output", required=True, help="Path to the output .apkg file.")
    parser.add_argument("--config", help="Path to a YAML, JSON, or TOML config file.")
    parser.add_argument("--topic", help="Override the topic for example generation.")
    parser.add_argument("--lesson", help="Override the lesson title.")
    parser.add_argument("--exam-level", help="Override the exam level label.")
    parser.add_argument("--force", action="store_true", help="Regenerate cards even if cache entries exist.")
    parser.add_argument("--log-level", help="Override the log level (DEBUG, INFO, WARNING, ERROR).")
    parser.add_argument("--parallelism", type=int, help="Maximum number of parallel LLM requests.")
    parser.add_argument("--base-url", help="Override the LLM chat-completions base URL.")
    parser.add_argument("--api-token", help="Override the LLM API token.")
    parser.add_argument("--model", help="Override the LLM model name.")
    parser.add_argument("--timeout", type=float, help="Override the LLM timeout in seconds.")
    parser.add_argument("--retries", type=int, help="Override the number of retry attempts.")
    parser.add_argument("--backoff", type=float, help="Override retry backoff seconds.")
    parser.add_argument("--temperature", type=float, help="Override the LLM temperature.")
    parser.add_argument("--max-tokens", type=int, help="Override the LLM max_tokens value.")
    parser.add_argument(
        "--header",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Add a custom HTTP header for the LLM request. Can be repeated.",
    )
    parser.add_argument("--deck-name", help="Override the Anki deck name.")
    parser.add_argument("--cache-dir", help="Override the local cache directory.")
    return parser


def parse_headers(values: list[str]) -> dict[str, str]:
    """Parse repeated KEY=VALUE header arguments."""
    headers: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"invalid header override '{value}'; expected KEY=VALUE")
        key, header_value = value.split("=", 1)
        key = key.strip()
        header_value = header_value.strip()
        if not key or not header_value:
            raise ValueError(f"invalid header override '{value}'; expected KEY=VALUE")
        headers[key] = header_value
    return headers


def build_overrides(args: argparse.Namespace) -> dict[str, Any]:
    """Translate CLI arguments into a settings override mapping."""
    llm_overrides: dict[str, Any] = {}
    if args.base_url:
        llm_overrides["base_url"] = args.base_url
    if args.api_token:
        llm_overrides["api_token"] = args.api_token
    if args.model:
        llm_overrides["model_name"] = args.model
    if args.timeout is not None:
        llm_overrides["timeout_seconds"] = args.timeout
    if args.retries is not None:
        llm_overrides["max_retries"] = args.retries
    if args.backoff is not None:
        llm_overrides["retry_backoff_seconds"] = args.backoff
    if args.temperature is not None:
        llm_overrides["temperature"] = args.temperature
    if args.max_tokens is not None:
        llm_overrides["max_tokens"] = args.max_tokens
    if args.header:
        llm_overrides["custom_headers"] = parse_headers(args.header)

    generation_overrides: dict[str, Any] = {}
    if args.topic:
        generation_overrides["default_topic"] = args.topic
    if args.lesson:
        generation_overrides["default_lesson"] = args.lesson
    if args.exam_level:
        generation_overrides["default_exam_level"] = args.exam_level
    if args.parallelism is not None:
        generation_overrides["parallelism"] = args.parallelism

    deck_overrides: dict[str, Any] = {}
    if args.deck_name:
        deck_overrides["deck_name"] = args.deck_name

    cache_overrides: dict[str, Any] = {}
    if args.cache_dir:
        cache_overrides["directory"] = args.cache_dir

    logging_overrides: dict[str, Any] = {}
    if args.log_level:
        logging_overrides["level"] = args.log_level

    overrides: dict[str, Any] = {}
    if llm_overrides:
        overrides["llm"] = llm_overrides
    if generation_overrides:
        overrides["generation"] = generation_overrides
    if deck_overrides:
        overrides["deck"] = deck_overrides
    if cache_overrides:
        overrides["cache"] = cache_overrides
    if logging_overrides:
        overrides["logging"] = logging_overrides
    return overrides


def configure_logging(settings: AppSettings) -> None:
    """Configure application logging."""
    logging.basicConfig(
        level=getattr(logging, settings.logging.level.upper(), logging.INFO),
        format="%(levelname)s %(name)s: %(message)s",
    )


def main(argv: list[str] | None = None) -> int:
    """Run the CLI command."""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        overrides = build_overrides(args)
        config_path = Path(args.config) if args.config else None
        settings = load_settings(config_path=config_path, overrides=overrides)
        configure_logging(settings)
    except Exception as exc:
        parser.error(str(exc))

    pipeline = DeckGenerationPipeline(settings)
    result = pipeline.run(
        input_path=Path(args.input),
        output_path=Path(args.output),
        topic=args.topic,
        lesson=args.lesson,
        exam_level=args.exam_level,
        force=args.force,
    )

    logging.info(
        "Deck written to %s (%s/%s cards, %s cache hits).",
        result.output_path,
        result.generated_items,
        result.total_items,
        result.cached_items,
    )

    if result.failed_items:
        summary = json.dumps([asdict(item) for item in result.failed_items], ensure_ascii=False, indent=2)
        logging.error("Some items failed and were skipped:\n%s", summary)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
