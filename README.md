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
- `lesson_topic`
- `form_examples`
- `tags`

POS-specific fields:
- countable nouns: article included directly in `dutch_word`, plus `plural_form`, `front_hint`, and `singular` / `plural` examples
- uncountable nouns: article included directly in `dutch_word`, plus `front_hint`, with `plural_form: null` and one `default` example
- verbs: `verb_forms`, with `present_tense` showing both `ik` and `hij` forms, plus `present_tense`, `past_tense`, and `past_participle` examples
- adjectives with two visible forms: `base_form` and `e_form` examples, with regular adjective form data kept out of `adjective_forms`
- adjectives without a distinct `-e` form: one `single_form` example in a context that clearly shows the missing `-e`, e.g. `de gouden ring`

### Sample JSON Object
```json
{
  "dutch_word": "de school",
  "russian_translation": "школа",
  "part_of_speech": "noun",
  "ipa_transcription": "sxoːl",
  "lesson_topic": "De school",
  "form_examples": [
    {
      "kind": "singular",
      "form": "de school",
      "example_sentence_nl": "De school is dichtbij.",
      "example_sentence_ru": "Школа находится рядом."
    },
    {
      "kind": "plural",
      "form": "scholen",
      "example_sentence_nl": "De scholen zijn dichtbij.",
      "example_sentence_ru": "Школы находятся рядом."
    }
  ],
  "tags": ["school", "lesson-3", "noun"],
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
- `Plural`
- `Plural_Audio`
- `Verb_Forms`
- `Adjective_Forms`
- `Word_Audio`
- `Example_1_Form`
- `Example_1_NL`
- `Example_1_RU`
- `Example_1_Audio`
- `Example_2_Form`
- `Example_2_NL`
- `Example_2_RU`
- `Example_2_Audio`
- `Example_3_Form`
- `Example_3_NL`
- `Example_3_RU`
- `Example_3_Audio`
- `Lesson`
- `Topic`
- `SourceWord`

Card behavior:
- Front side is Russian-driven.
- Countable noun cards explicitly prompt plural recall.
- Uncountable noun cards keep the front hint plain and do not add `(множественное число?)`.
- Regular adjective cards do not list predictable endings as grammar fields, but examples must show both visible forms, e.g. `mooi` and `mooie`.
- Onverbuigbare adjectives use one clear `single_form` example in a context where regular adjectives would normally take `-e`, e.g. `de gouden ring`.
- Back side shows Dutch, IPA, grammar details, then the generated examples. Each example shows the form label, Russian sentence, Dutch sentence, and its matching audio reference when available.
- `Word_Audio`, `Plural_Audio`, and per-example audio fields are populated with packaged `[sound:...]` references when audio generation is enabled.

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
- generates one MP3 for `Plural_Audio` when a plural form exists
- generates one MP3 per populated example slot
- reuses existing files in `audio.directory` for unchanged text, voice, and output format
- writes Anki `[sound:...]` references into the note fields and bundles the media into the `.apkg`

Audio failures are non-fatal for deck writing. The affected sound reference is omitted, the deck is still written for successful cards, and the CLI exits with a non-zero status.

## Testing
Run the test suite with:

```bash
source /Users/dstafichuk/setup_env.sh && .venv/bin/python -m pytest -q
```

Included tests cover:
- schema validation
- LLM response parsing
- deck generation
- pipeline caching and partial-failure behavior

## Notes
- The client targets OpenAI-compatible chat-completions APIs.
- If some items fail validation after all retries, the deck is still written for successful items and the CLI exits with a non-zero status.
