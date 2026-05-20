# HSCode Classifier

A **production-ready, model-agnostic** Python library for classifying product descriptions into 8-digit EU Combined Nomenclature (CN) / Harmonized System (HS) codes.

Given a free-text product description, the library returns the most appropriate **8-digit CN code** along with a confidence score, the official EU description, and the required supplementary unit (e.g. `PST`, `M2`, `KG_NET_EDA`).

## Key features

- **Model agnostic** — works with OpenAI, Azure OpenAI, Anthropic, Google Gemini, Ollama (local), OpenRouter (hundreds of models behind one API), or any custom LLM by implementing a simple interface.
- **Hierarchical classification with backtracking** — traverses the official HS tree (Chapter → Heading → Subheading → CN code) and backtracks if an LLM detects a wrong turn upstream.
- **Always-fresh nomenclature** — fetches CN codes directly from the EU Publications Office [SPARQL endpoint](https://op.europa.eu/en/web/eu-vocabularies). Cached locally per year+month to avoid repeated API calls.
- **Regex fast-path** — if the description already contains a valid 8-digit code (`"... CN 39269097 ..."`), it's extracted and validated without any LLM call.
- **Validated output** — every returned code is validated against the official EU CN database.
- **Supplementary unit lookup** — automatically attaches the statistical unit (pieces, m², litres …) for Intrastat reporting.
- **No secrets in code** — all credentials come from environment variables.
- **Production-ready** — retries with exponential backoff, structured logging, typed result objects, full test suite, Dockerfile.

## Installation

```bash
pip install -e .
```

Or with a specific provider:

```bash
pip install -e ".[openai]"      # OpenAI
pip install -e ".[azure]"       # Azure OpenAI
pip install -e ".[openrouter]"  # OpenRouter (uses the OpenAI SDK)
pip install -e ".[anthropic]"   # Anthropic Claude
pip install -e ".[google]"      # Google Gemini
pip install -e ".[ollama]"      # Local Ollama
pip install -e ".[all]"         # Everything
```

## Quick start

```python
from hscode import classify

result = classify(
    description="Wireless bluetooth headphones with active noise cancelling",
    provider="openai",                # or "azure", "anthropic", "google", "ollama", "openrouter"
    model="gpt-5.4-nano",             # any model your provider supports
)

print(result.hs_code)           # "85183000"
print(result.description)       # "Headphones and earphones, ..."
print(result.confidence)        # 0.92
print(result.supplementary_unit)# "PST"
print(result.reasoning)         # full chain-of-thought
```

Credentials come from the environment (no secrets in code):

```bash
export OPENAI_API_KEY=sk-...
```

### Using the classifier directly (re-use the LLM client)

```python
from hscode import HSCodeClassifier
from hscode.providers import OpenAIProvider

classifier = HSCodeClassifier(provider=OpenAIProvider(model="gpt-5.4-nano"))

for desc in ["Cotton T-shirt", "Steel screws M6", "Lithium-ion battery 18650"]:
    result = classifier.classify(desc)
    print(f"{desc:<45} -> {result.hs_code} ({result.confidence:.0%})")
```

### CLI

```bash
hscode "Wireless bluetooth headphones" --provider openai --model gpt-5.4-nano
```

JSON output:

```bash
hscode "Wireless bluetooth headphones" --json
```

### OpenRouter — pick from hundreds of models

```bash
export OPENROUTER_API_KEY=sk-or-...

# Use any vendor/model slug supported by OpenRouter:
hscode "Lithium-ion battery 18650" \
    --provider openrouter \
    --model anthropic/claude-sonnet-4.5

# Discover models that support structured (schema-constrained) output:
hscode --list-openrouter-models
```

Programmatic discovery:

```python
from hscode.providers.openrouter_provider import OpenRouterProvider

# Full live catalogue (357+ models at the time of writing):
all_models = OpenRouterProvider.list_models()

# Only the ~280 models that support response_format / structured_outputs
# (recommended for HSCode, which relies on JSON-schema-constrained output):
suitable = OpenRouterProvider.list_models(structured_only=True)

for m in suitable[:5]:
    print(m["id"], m.get("context_length"), m["pricing"])
```

## Configuration

All configuration is via environment variables. **No secret is ever read from a file in the repo.**

| Variable | Description |
|---|---|
| `HSCODE_PROVIDER` | Default provider (`openai`, `azure`, `anthropic`, `google`, `ollama`, `openrouter`) |
| `HSCODE_MODEL` | Default model name |
| `HSCODE_CACHE_DIR` | Where to cache CN data (default: `~/.cache/hscode`) |
| `HSCODE_CN_YEAR` | Force a specific CN year (default: auto) |
| `HSCODE_MAX_RETRIES` | Max backtracking attempts (default: 3) |
| `HSCODE_LOG_LEVEL` | Logging level (default: `INFO`) |
| `OPENAI_API_KEY` | For the OpenAI provider |
| `AZURE_OPENAI_API_KEY` / `AZURE_OPENAI_ENDPOINT` / `AZURE_OPENAI_DEPLOYMENT` | For Azure OpenAI |
| `ANTHROPIC_API_KEY` | For Anthropic |
| `GOOGLE_API_KEY` | For Google Gemini |
| `OLLAMA_HOST` | For local Ollama (default: `http://localhost:11434`) |
| `OPENROUTER_API_KEY` | For OpenRouter (use a fully-qualified model slug like `openai/gpt-5.4-nano` or `anthropic/claude-haiku-4.5`) |
| `OPENROUTER_HTTP_REFERER` / `OPENROUTER_APP_TITLE` | Optional ranking headers for openrouter.ai/rankings |

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│                       classify(description)                │
└────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────┐
│  1. Regex fast-path: is there a valid 8-digit code in text?│
└────────────────────────────────────────────────────────────┘
                              │ (miss)
                              ▼
┌────────────────────────────────────────────────────────────┐
│  2. Hierarchical LLM classification with backtracking      │
│     Chapter (2) → Heading (4) → Subheading (6) → CN (8)    │
└────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────┐
│  3. Validate against EU CN database (SPARQL)               │
│     Attach official description + supplementary unit       │
└────────────────────────────────────────────────────────────┘
```

The LLM never sees the full nomenclature at once — only the relevant subset for the current level. This keeps prompts small, focused, and cheap.

## Adding a custom LLM provider

Implement a single method:

```python
from hscode.providers import LLMProvider, StructuredOutput

class MyProvider(LLMProvider):
    def generate_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: type[StructuredOutput],
    ) -> StructuredOutput:
        # Call your model and return an instance of `schema`
        ...
```

Pass it directly to the classifier:

```python
HSCodeClassifier(provider=MyProvider())
```

## Testing

```bash
pip install -e ".[dev]"
pytest
```

## Docker

```bash
docker build -t hscode .
docker run --rm -e OPENAI_API_KEY=$OPENAI_API_KEY hscode \
    "Wireless bluetooth headphones"
```

## License

MIT
