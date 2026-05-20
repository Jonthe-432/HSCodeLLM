"""Command-line interface for HSCode.

Examples
--------

::

    hscode "Wireless bluetooth headphones with ANC"
    hscode "Cotton T-shirt" --provider anthropic --model claude-haiku-4.5
    hscode "Steel screws M6 zinc plated" --json
    hscode --preload-cache
    hscode --list-openrouter-models
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

from hscode import __version__
from hscode.api import classify
from hscode.cn_retriever import preload_cn_data
from hscode.llm import list_providers
from hscode.logging_config import configure_logging, get_logger

logger = get_logger("cli")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hscode",
        description=(
            "Classify a product description into an 8-digit EU Combined "
            "Nomenclature (CN) / Harmonized System (HS) code."
        ),
    )
    parser.add_argument(
        "description",
        nargs="?",
        help="Product description to classify (omit when using --preload-cache or --list-openrouter-models).",
    )
    parser.add_argument(
        "--provider",
        choices=list_providers(),
        help="LLM provider (default: $HSCODE_PROVIDER or 'openai').",
    )
    parser.add_argument(
        "--model",
        help="Model slug passed to the provider (default: $HSCODE_MODEL).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Sampling temperature (default: 0.0).",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=None,
        help="Maximum number of hierarchical passes (default: $HSCODE_MAX_RETRIES or 3).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output a JSON object instead of human-readable text.",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress log output.",
    )
    parser.add_argument(
        "--preload-cache",
        action="store_true",
        help="Pre-load CN data from SPARQL into the local cache and exit.",
    )
    parser.add_argument(
        "--list-openrouter-models",
        action="store_true",
        help=(
            "Print available OpenRouter model slugs (requires "
            "OPENROUTER_API_KEY) and exit. Filtered to models that support "
            "structured outputs."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"hscode {__version__}",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    configure_logging("ERROR" if args.quiet else None)

    if args.preload_cache:
        ok = preload_cn_data()
        if ok:
            print("CN data successfully cached.", file=sys.stderr)
            return 0
        print("Failed to preload CN data.", file=sys.stderr)
        return 1

    if args.list_openrouter_models:
        return _print_openrouter_models(as_json=args.json)

    if not args.description:
        parser.error("description is required (or use --preload-cache / --list-openrouter-models)")

    extra_kwargs = {}
    if args.temperature is not None:
        extra_kwargs["temperature"] = args.temperature

    try:
        result = classify(
            args.description,
            provider=args.provider,
            model=args.model,
            max_retries=args.max_retries,
            **extra_kwargs,
        )
    except Exception as exc:
        logger.error("Classification failed: %s", exc)
        if args.json:
            print(json.dumps({"error": str(exc)}))
        else:
            print(f"Error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(result.model_dump_json(indent=2))
    else:
        print_human(result)
    return 0


def _print_openrouter_models(as_json: bool = False) -> int:
    """Print the OpenRouter model catalogue (structured-output capable)."""
    try:
        from hscode.openrouter import list_models, OpenRouterError
    except ImportError as exc:  # pragma: no cover
        print(f"OpenRouter helper unavailable: {exc}", file=sys.stderr)
        return 2

    try:
        models = list_models(structured_only=True)
    except OpenRouterError as exc:
        print(f"Could not fetch OpenRouter models: {exc}", file=sys.stderr)
        return 2

    if as_json:
        slim = [
            {
                "id": m.get("id"),
                "name": m.get("name"),
                "context_length": m.get("context_length"),
                "pricing": m.get("pricing"),
            }
            for m in models
        ]
        print(json.dumps(slim, indent=2))
        return 0

    print(f"# {len(models)} OpenRouter models supporting structured outputs")
    for m in sorted(models, key=lambda x: x.get("id", "")):
        ctx = m.get("context_length") or "?"
        pricing = m.get("pricing") or {}
        prompt = pricing.get("prompt", "?")
        completion = pricing.get("completion", "?")
        print(f"  {m['id']:<55}  ctx={ctx:>8}  prompt=${prompt}/tok  completion=${completion}/tok")
    return 0


def print_human(result) -> None:
    """Pretty-print a ClassificationResult for terminal users."""
    print(f"HS code:              {result.hs_code}")
    print(f"Description:          {result.description}")
    print(f"Confidence:           {result.confidence:.0%}")
    if result.supplementary_unit:
        print(
            f"Supplementary unit:   {result.supplementary_unit}"
            f" ({result.supplementary_unit_description})"
        )
    if result.chapter:
        print(f"Path:                 {result.chapter} → {result.heading} → {result.subheading} → {result.hs_code}")
    print(f"Status:               {result.status}")
    print(f"Validated:            {'yes' if result.validated else 'no'}")
    print(f"Attempts:             {result.attempts}")
    if result.cn_year:
        print(f"CN year:              {result.cn_year}")
    print("")
    print("Reasoning:")
    for line in result.reasoning.splitlines():
        print(f"  {line}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
