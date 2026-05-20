# HSCode Classifier

A **production-ready, model-agnostic** Python library for classifying product descriptions into 8-digit EU Combined Nomenclature (CN) / Harmonized System (HS) codes.

Given a free-text product description, the library returns the most appropriate **8-digit CN code** along with a confidence score, the official EU description, and the required supplementary unit (e.g. `PST`, `M2`, `KG_NET_EDA`).

## Key features

- **LangChain under the hood** — every provider (OpenAI, Azure OpenAI, Anthropic, Google Gemini, Ollama, OpenRouter) is a LangChain `BaseChatModel`. Swapping the model is a one-line change; no provider-specific glue lives in this codebase.
- **Conversational classification** — the entire hierarchical walk (Chapter → Heading → Subheading → CN code) happens inside one ongoing chat, so the model remembers its own prior choices and reasoning. This dramatically reduces the "model keeps re-picking the wrong chapter" failure mode.
- **Hierarchical traversal with backtracking** — at any level the model can say `BACKTRACK` and the classifier rewinds, while preserving conversation history.
- **Always-fresh nomenclature** — fetches CN codes directly from the EU Publications Office [SPARQL endpoint](https://op.europa.eu/en/web/eu-vocabularies). Cached locally per year+month.
- **Robust to flat headings** — ~33% of EU CN headings have no level-6 subheading rows in the SPARQL data. The retriever synthesises them from level-8 prefixes so the conversation can still narrow down.
- **Regex fast-path** — if the description already contains a valid 8-digit code (`"... CN 39269097 ..."`), it's extracted and validated without any LLM call.
- **Validated output** — every returned code is validated against the official EU CN database.
- **Supplementary unit lookup** — automatically attaches the statistical unit (pieces, m², litres …) for Intrastat reporting.
- **No secrets in code** — all credentials come from environment variables.

## Installation

```bash
pip install -e .
```

Pick one or more provider extras:

```bash
pip install -e ".[openai]"      # OpenAI (langchain-openai)
pip install -e ".[azure]"       # Azure OpenAI (langchain-openai)
pip install -e ".[openrouter]"  # OpenRouter (langchain-openrouter)
pip install -e ".[anthropic]"   # Anthropic Claude (langchain-anthropic)
pip install -e ".[google]"      # Google Gemini (langchain-google-genai)
pip install -e ".[ollama]"      # Local Ollama (langchain-ollama)
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
print(result.confidence)        # 0.93
print(result.supplementary_unit)# "NO_SU"
print(result.reasoning)         # full chain-of-thought
```

Credentials come from the environment (no secrets in code):

```bash
export OPENAI_API_KEY=sk-...
```

### Using the classifier directly (re-use the LLM client)

```python
from hscode import HSCodeClassifier, get_chat_model

llm = get_chat_model(provider="openai", model="gpt-5.4-nano")
classifier = HSCodeClassifier(llm=llm)

for desc in ["Cotton T-shirt", "Steel screws M6", "Lithium-ion battery 18650"]:
    result = classifier.classify(desc)
    print(f"{desc:<45} -> {result.hs_code} ({result.confidence:.0%})")
```

You can also pass any LangChain `BaseChatModel` directly:

```python
from langchain.chat_models import init_chat_model
from hscode import HSCodeClassifier

llm = init_chat_model("openrouter:anthropic/claude-haiku-4.5", temperature=0.0)
classifier = HSCodeClassifier(llm=llm)
result = classifier.classify("Wireless bluetooth headphones")
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
    --model anthropic/claude-haiku-4.5

# Discover models that support structured (schema-constrained) output:
hscode --list-openrouter-models
```

Programmatic discovery:

```python
from hscode.openrouter import list_models

# Full live catalogue (357+ models at the time of writing):
all_models = list_models()

# Only the ~280 models that support response_format / structured_outputs
# (recommended for HSCode, which relies on JSON-schema-constrained output):
suitable = list_models(structured_only=True)

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
│  2. Conversational hierarchical classification             │
│     (one chat, one LangChain BaseChatModel)                │
│     Chapter (2) → Heading (4) → Subheading (6) → CN (8)    │
│     Model can BACKTRACK at any level; conversation history │
│     is preserved across the whole walk.                    │
└────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────┐
│  3. Validate against EU CN database (SPARQL)               │
│     Attach official description + supplementary unit       │
└────────────────────────────────────────────────────────────┘
```

The LLM never sees the full nomenclature at once — only the relevant subset for the current level. This keeps prompts small, focused, and cheap.

## Using a custom LLM

Any LangChain `BaseChatModel` will work — anything that supports
`with_structured_output(PydanticSchema)`:

```python
from langchain_openai import ChatOpenAI       # or any other langchain-* package
from hscode import HSCodeClassifier

llm = ChatOpenAI(model="gpt-5.4-nano", temperature=0.0)
classifier = HSCodeClassifier(llm=llm)
result = classifier.classify("Stainless steel kitchen knife")
```

For a list of officially-supported chat models see https://docs.langchain.com/oss/python/integrations/chat/.

## Testing

```bash
pip install -e ".[dev]"
pytest
```

Tests use a scripted LangChain chat model (`tests/conftest.py::ScriptedChatModel`) — no network calls, no API keys needed.

## Docker

```bash
docker build -t hscode .
docker run --rm -e OPENAI_API_KEY=$OPENAI_API_KEY hscode \
    "Wireless bluetooth headphones"
```

## License

MIT
