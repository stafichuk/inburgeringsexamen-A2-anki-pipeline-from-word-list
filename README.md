# Dutch A2 Inburgering Anki Pipeline

CLI application for generating `.apkg` Anki decks from plain Dutch word lists. The tool is designed for Russian-speaking learners preparing for the A2 Inburgering Spreken exam and can optionally generate Dutch audio with Azure Text to Speech.

## Features
- Reads a plain text file with one Dutch word or phrase per line.
- Calls an external OpenAI-compatible LLM endpoint to infer part of speech and generate structured card data.
- Validates every model response against strict Pydantic schemas.
- Retries malformed responses and fails per-item rather than losing the whole run.
- Caches generated items locally by word, topic, lesson, model, and prompt version.
- Generates uncached items in parallel with a bounded worker pool.
- Generates a real `.apkg` deck with a custom note type using `genanki`.
- Optionally generates and packages Dutch word and example-sentence audio.

## Project Structure
```text
app/
  anki.py
  audio.py
  cache.py
  cli.py
  config.py
  llm_client.py
  models.py
  pipeline.py
  prompts.py
tests/
config.example.yaml
words.example.txt
pyproject.toml
README.md
```

## Requirements
- Python 3.11+
- A reachable OpenAI-compatible chat-completions endpoint
- An Azure Speech resource if `audio.enabled` is true

## Installation
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
```

## Configuration
The app accepts YAML, JSON, or TOML config files. YAML is the primary format.

Example config:

```yaml
llm:
  base_url: "https://example-llm-provider.local/v1/chat/completions"
  api_token: "replace-me"
  model_name: "gpt-4o-mini"
  custom_headers:
    X-Client-Name: "dutch-a2-anki-pipeline"
  timeout_seconds: 60
  max_retries: 2
  retry_backoff_seconds: 1.5
  temperature: 0.2
  max_tokens: 800

deck:
  deck_name: "Lesson 3 - De school"

generation:
  default_topic: "De school"
  default_lesson: "Lesson 3"
  default_exam_level: "A2 Inburgering Spreken"
  parallelism: 4

cache:
  directory: ".cache/cards"

audio:
  enabled: false
  provider: "azure"
  directory: ".cache/audio"
  azure:
    region: "westeurope"
    # endpoint: "https://westeurope.tts.speech.microsoft.com/cognitiveservices/v1"
    api_key: "replace-me"
    voice: "nl-NL-FennaNeural"

logging:
  level: "INFO"
```

See [`config.example.yaml`](config.example.yaml).

## Usage
Generate a deck with config defaults:

```bash
generate-deck \
  --input words.example.txt \
  --output school.apkg \
  --config config.example.yaml
```

Override topic and lesson at runtime:

```bash
generate-deck \
  --input words.example.txt \
  --output school.apkg \
  --config config.example.yaml \
  --parallelism 6 \
  --topic "De school" \
  --lesson "Lesson 3"
```

Force regeneration and bypass cache:

```bash
generate-deck \
  --input words.example.txt \
  --output school.apkg \
  --config config.example.yaml \
  --force
```

Use direct CLI overrides without a config file:

```bash
generate-deck \
  --input words.example.txt \
  --output school.apkg \
  --base-url "https://provider.example/v1/chat/completions" \
  --api-token "replace-me" \
  --model "gpt-4o-mini" \
  --topic "De school" \
  --lesson "Lesson 3"
```

## LLM Output Schema
The model is prompted to return JSON only. Responses are validated against strict Pydantic models with POS-specific requirements.

Core fields:
- `dutch_word`
- `russian_translation`
- `part_of_speech`
- `ipa_transcription`
- `example_sentence_nl`
- `example_sentence_ru`
- `lesson_topic`
- `tags`

POS-specific fields:
- nouns: `article`, `plural_form`, `front_hint`
- verbs: `verb_forms`
- adjectives: `adjective_forms` only for onverbuigbare adjectives; regular adjectives use `null`

### Sample JSON Object
```json
{
  "dutch_word": "school",
  "russian_translation": "школа",
  "part_of_speech": "noun",
  "ipa_transcription": "sxoːl",
  "example_sentence_nl": "Mijn school is dichtbij.",
  "example_sentence_ru": "Моя школа находится рядом.",
  "lesson_topic": "De school",
  "tags": ["school", "lesson-3", "noun"],
  "article": "de",
  "plural_form": "scholen",
  "front_hint": "школа (множественное число?)",
  "verb_forms": null,
  "adjective_forms": null
}
```

## Anki Note Design
The generated note type contains these fields:

- `Front`
- `Word_NL`
- `Translation_RU`
- `IPA`
- `POS`
- `Article`
- `Plural`
- `Verb_Forms`
- `Adjective_Forms`
- `Example_NL`
- `Example_RU`
- `Word_Audio`
- `Example_Audio`
- `Lesson`
- `Topic`
- `SourceWord`

Card behavior:
- Front side is Russian-driven.
- Noun cards explicitly prompt plural recall.
- Regular adjective cards do not list predictable adjective endings.
- Onverbuigbare adjectives show a short note and example phrase, such as `gouden ring`.
- Back side shows Dutch, IPA, grammar details, then the example sentence in this order:
  1. Russian translation
  2. Dutch sentence
- `Word_Audio` and `Example_Audio` are populated with packaged `[sound:...]` references when audio generation is enabled.

## Caching
Each successful generation is cached locally in `.cache/cards/` by:
- normalized source word
- topic
- lesson
- exam level
- model name
- prompt version

Use `--force` to bypass cache reads and overwrite cached entries.

## Parallelism
Uncached words are generated concurrently. By default, the pipeline uses up to `4` parallel requests.

Tune this with config:

```yaml
generation:
  parallelism: 6
```

Or override it on the CLI:

```bash
generate-deck --input words.txt --output school.apkg --config config.yaml --parallelism 6
```

For this workload, bounded parallelism is a better fit than batching:
- cache entries remain one word per file
- one bad response only affects one word
- retries stay per item
- deck output still preserves the original input order

## Audio Generation
Audio generation is disabled by default. Enable it in the config file:

```yaml
audio:
  enabled: true
  provider: "azure"
  directory: ".cache/audio"
  azure:
    region: "westeurope"
    api_key: "replace-me"
    voice: "nl-NL-FennaNeural"
```

You can provide `audio.azure.endpoint` instead of `audio.azure.region` when you need to target a specific Azure Speech endpoint. If both are set, the explicit endpoint is used.

When enabled, the app:
- generates one MP3 for `Word_Audio` from the Dutch word
- generates one MP3 for `Example_Audio` from the Dutch example sentence
- reuses existing files in `audio.directory` for unchanged text, voice, and output format
- writes Anki `[sound:...]` references into the note fields and bundles the media into the `.apkg`

Audio failures are non-fatal for deck writing. The affected sound reference is omitted, the deck is still written for successful cards, and the CLI exits with a non-zero status.

## Testing
Run the test suite with:

```bash
pytest
```

Included tests cover:
- schema validation
- LLM response parsing
- deck generation
- pipeline caching and partial-failure behavior

## Notes
- The client targets OpenAI-compatible chat-completions APIs.
- If some items fail validation after all retries, the deck is still written for successful items and the CLI exits with a non-zero status.
