# Dutch A2 Inburgering Anki Pipeline

CLI application for generating `.apkg` Anki decks from plain Dutch word lists or assembling them from pre-generated card data. The tool is designed for Russian-speaking learners preparing for the A2 Inburgering Spreken exam and can optionally generate Dutch audio with Azure Text to Speech.

## Features
- Reads a plain text file with one Dutch word or phrase per line and optional stable entry IDs.
- Calls an external OpenAI-compatible LLM endpoint to infer part of speech and generate structured card data.
- Provides an assemble-only mode that validates prepared card data, generates audio, and writes the deck without using an LLM or generated-card cache.
- Supports optional per-line Russian translation hints for sense-specific cards.
- Supports explicit ` | `-separated Dutch answers for one Russian active-recall concept.
- Validates every model response against strict Pydantic schemas.
- Generates all pending cache misses together so the model can coordinate varied examples across the deck.
- Uses accepted cards as immutable diversity context when generating newly added or previously failed items.
- Caches every valid card before retrying only missing or invalid cards.
- Publishes the `.apkg` only when every current input entry has a valid card.
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
  generated_input.py
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
- A reachable OpenAI-compatible chat-completions endpoint for word-list generation mode
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
  max_tokens: 65536

deck:
  deck_name: "Lesson 3 - De school"

generation:
  default_topic: "De school"
  default_lesson: "Lesson 3"
  default_exam_level: "A2 Inburgering Spreken"

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

The `llm` section is required in the default `generate` mode. It may be omitted in `assemble` mode; deck, generation-default, audio, and logging settings remain available.

## Usage
Generate a deck with config defaults:

```bash
generate-deck \
  --input words.txt \
  --output formulier.apkg \
  --config config.yaml
```

Override topic and lesson at runtime:

```bash
generate-deck \
  --input words.example.txt \
  --output school.apkg \
  --config config.example.yaml \
  --topic "De school" \
  --lesson "Lesson 3"
```

Start or resume a coordinated full refresh:

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

Assemble a deck from a prepared generated-data manifest without calling an LLM:

```bash
generate-deck \
  --mode assemble \
  --input generated-cards.json \
  --output school.apkg \
  --config audio-and-deck.yaml
```

Assemble mode does not read or write `.cache/cards`. It validates the complete manifest first, then generates any enabled audio and writes the deck. Generation-only options such as `--force`, LLM overrides, and `--cache-dir` are rejected.

## Input Word List
Each non-comment line is a Dutch item or an explicit group of accepted Dutch answers, with an optional stable ID and optional strict Russian translation hint:

```text
de broer
[neef-nephew] de neef - племянник
[neef-cousin] de neef - двоюродный брат
[clothes] de kleding | de kleren - одежда
[friend] de vrient
kinderopvang - детский сад
e-mail
```

The optional `[id]` prefix is the stable identity of the learner-facing concept. When it is omitted, identity is derived from the normalized first Dutch answer plus its translation hint, if present. Ordinary words and same-spelling words with different hints therefore stay as easy to author as before. Use an explicit ID when you want to correct or reorder answers later without changing the Anki note identity, or when two lines would otherwise have the same implicit identity. For example, `[friend] de vrient` can later become `[friend] de vriend` while keeping the same note identity and Anki scheduling. IDs must be unique within the input.

The translation delimiter is the spaced form ` - `. Hyphenated Dutch words such as `e-mail` are treated as plain items. Different hints already give the two `de neef` lines separate implicit identities; their explicit IDs additionally keep those identities stable if a word or hint is corrected later.

The accepted-answer delimiter is the exact spaced form ` | ` and is used only on the Dutch side. A grouped line must include a Russian hint, which becomes its authoritative front. The alternatives are explicit: the model generates data for every listed answer but may not add, replace, or merge answers. The authored order is preserved.

```text
de kleding | de kleren - одежда
```

This produces one Anki card and one schedule. The front is `одежда`; recalling either Dutch answer counts as correct. The back presents both answers equally, each with its own pronunciation, audio, grammar, and examples. The front does not add a plural-recall question because grouped answers can have different number behaviour.

## Generated-Data Input

Assemble mode accepts strict JSON with a fixed format marker, schema version, and an ordered list of learner-facing concepts. Each answer contains the exact authored Dutch input and one complete card using the `GeneratedCard` schema documented below.

```json
{
  "format": "dutch-a2-generated-cards",
  "schema_version": 1,
  "concepts": [
    {
      "entry_id": "yesterday",
      "translation_hint": "вчера",
      "topic": "Tijd",
      "lesson": "Les 1",
      "exam_level": "A2 Inburgering Spreken",
      "answers": [
        {
          "input_item": "gisteren",
          "card": {
            "dutch_word": "gisteren",
            "russian_translation": "вчера",
            "part_of_speech": "adverb",
            "ipa_transcription": "ˈɣɪstərə(n)",
            "lesson_topic": "Tijd",
            "form_examples": [
              {
                "kind": "default",
                "form": "gisteren",
                "example_sentence_nl": "Gisteren werkte ik thuis.",
                "example_sentence_ru": "Вчера я работал дома."
              }
            ],
            "tags": ["tijd"]
          }
        }
      ]
    }
  ]
}
```

Concept and answer order is preserved. For a grouped note, place each accepted Dutch answer in the same concept's `answers` array; array position becomes its stable answer position. `entry_id`, `translation_hint`, `topic`, `lesson`, and `exam_level` retain the source identity and deck metadata that raw LLM card objects do not contain. Context fields may be omitted and filled from generation defaults, or explicitly overridden with `--topic`, `--lesson`, and `--exam-level`.

Unknown fields, unsupported schema versions, empty concepts, invalid generated cards, duplicate source identities, and grouped cards that replace an authored Dutch answer are rejected before TTS or deck writing begins.

## LLM Output Schema
The model is prompted to return JSON only. Each batch row must echo its `source_id`, exact `input_item`, and exact `translation_hint` (including `null`), which prevents a valid card or same-word sense from being accepted under a swapped source row. Nested card responses are validated against strict Pydantic models with POS-specific requirements.

Core fields:
- `dutch_word`
- `russian_translation`
- `part_of_speech`
- `ipa_transcription`
- `lesson_topic`
- `form_examples`
- `tags`
- `noun_number` for nouns

POS-specific fields:
- countable nouns: article included directly in `dutch_word`, bare `plural_form`, `front_hint`, and `singular` / `plural` examples whose `form` value appears exactly in the sentence
- uncountable nouns: article included directly in `dutch_word`, plus `front_hint`, with `plural_form: null` and one `default` example whose `form` value appears exactly in the sentence
- plural-only nouns: the plural headword retained in `dutch_word`, `noun_number: "plural_only"`, `plural_form: null`, and one `default` example
- verbs: `verb_forms`, with editable forms for `infinitive`, `present_ik`, `present_hij`, `past_tense`, and `perfect_tense`, plus `present_tense`, `past_tense`, and `perfect_tense` examples
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
      "form": "school",
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
  "noun_number": "countable",
  "plural_form": "scholen",
  "front_hint": "школа",
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
- `Verb_Infinitive`
- `Verb_Infinitive_Audio`
- `Verb_Present_Ik`
- `Verb_Present_Ik_Audio`
- `Verb_Present_Hij`
- `Verb_Present_Hij_Audio`
- `Verb_Past`
- `Verb_Past_Audio`
- `Verb_Perfect`
- `Verb_Perfect_Audio`
- `Verb_Notes`
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
- Grouped cards use the exact shared Russian hint on the front, accept recall of any listed answer, and render one complete back-side block per answer without a plural-recall prompt.
- Regular adjective cards do not list predictable endings as grammar fields, but examples must show both visible forms, e.g. `mooi` and `mooie`.
- Onverbuigbare adjectives use one clear `single_form` example in a context where regular adjectives would normally take `-e`, e.g. `de gouden ring`.
- Verb cards store each form in its own field, e.g. `leren`, `ik leer`, `hij leert`, `leerde`, and `heeft geleerd`, with paired audio fields so HyperTTS can regenerate individual form audio after manual corrections. The back-side grammar block shows the conjugated forms; the infinitive is already shown as the Dutch headword at the top.
- Back side shows Dutch, IPA, grammar details, then the generated examples. Each example shows the form label, Russian sentence, Dutch sentence, and its matching audio reference when available.
- `Word_Audio`, `Plural_Audio`, verb-form audio fields, and per-example audio fields are populated with packaged `[sound:...]` references when audio generation is enabled.

## Caching
Each accepted card is cached locally in `.cache/cards/`. Its reusable identity includes:

- the entry ID;
- the current source word and translation hint;
- topic;
- lesson;
- exam level.

Changing the source word or translation hint, topic, lesson, or exam level invalidates the affected accepted card. Changing the LLM model or prompt version does not invalidate accepted cards: model and prompt information is generation metadata, while previously accepted content remains frozen.

Grouped concepts keep one cache entry per Dutch answer. Appending a new ` | ` alternative therefore reuses every unchanged accepted answer and generates only the new one. A grouped card is not published until every explicitly listed answer has valid generated data.

An explicit entry ID separates Anki note identity from source content. Correcting the source of `[friend] de vrient` to `[friend] de vriend` regenerates its cached content because the source changed, but retains the stable ID used for the Anki note.

Assemble mode intentionally bypasses this generated-card cache because its input manifest is authoritative. The independent audio cache is still reused.

## Incremental Batch Generation

On the first run, all input entries are cache misses and are generated in one coordinated batch. On later runs, only new, changed, missing, or previously invalid entries are requested together. The Dutch examples from all accepted cached cards are included as immutable context so the model can avoid repeating their situations and sentence patterns; cached cards are not returned or rewritten by a normal run.

If a batch contains a mixture of valid and invalid or missing cards, every valid card is cached as soon as it is accepted. Automatic retries request only the unresolved entries and include the newly accepted cards in their diversity context. A later command continues in the same way if the retry limit is reached or the earlier command is interrupted.

The pipeline does not publish a card-incomplete deck. The requested output file, including any previously complete `.apkg` at that path, remains untouched until all current input entries have valid cards. Partial cache progress is retained even when no deck can yet be written.

`--force` starts a resumable full refresh of every current input entry. A durable refresh marker suppresses all previous entries before generation begins, so an interruption cannot mix refreshed cards with old fallbacks. Refreshed valid cards replace their previous cache entries as soon as they are accepted. If the refresh remains incomplete, a later normal run requests only the unresolved entries and uses the refreshed accepted cards as context. The previous complete `.apkg` remains untouched until the refresh is complete, and final package replacement is atomic.

This cache format is intentionally separate from the legacy per-card generation cache. The first run after upgrading therefore performs one complete coordinated regeneration. LLM generation no longer accepts the `generation.parallelism` setting or `--parallelism` option because pending cards share one batch request. Audio generation remains sequential and was never controlled by that setting.

Removing a line from the input omits that note from the next generated `.apkg`. Importing the replacement package into Anki does not delete a note that is already present in the collection; remove such notes manually in Anki.

Likewise, converting previously separate notes into one grouped concept does not delete the old standalone notes from an existing Anki collection. After importing the grouped deck, remove any obsolete standalone sibling notes manually once.

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
- generates one MP3 for each populated verb form field
- generates one MP3 per populated example slot
- reuses existing files in `audio.directory` for unchanged text, voice, and output format
- writes Anki `[sound:...]` references into the note fields and bundles the media into the `.apkg`

Audio failures are non-fatal for deck writing. All cards are still written, only the affected sound references are omitted, and the CLI exits with a non-zero status.

## Testing
Run the test suite with:

```bash
source /Users/dstafichuk/setup_env.sh && .venv/bin/python -m pytest -q
```

Included tests cover:
- schema validation
- LLM response parsing
- deck generation
- assemble-only generated-data input and CLI mode selection
- incremental batch caching, retry, and incomplete-deck behavior

## Notes
- The client targets OpenAI-compatible chat-completions APIs.
- If some entries remain unresolved after all retries, their valid batch peers stay cached, no incomplete deck is written, and the CLI exits with a non-zero status.
