"""Basic usage example for the HSCode classifier.

Run with:

    OPENAI_API_KEY=... python examples/basic_usage.py
"""

from __future__ import annotations

from hscode import HSCodeClassifier, classify, get_chat_model


def main() -> None:
    # ------------------------------------------------------------------
    # 1. One-shot classification (builds a chat model from env vars).
    # ------------------------------------------------------------------
    result = classify(
        "Wireless bluetooth headphones with active noise cancelling",
        # provider="openai",     # defaults to $HSCODE_PROVIDER
        # model="gpt-5.4-nano",  # defaults to $HSCODE_MODEL
    )
    print(f"{result.hs_code}  ({result.confidence:.0%})  {result.description}")
    if result.supplementary_unit:
        print(
            f"  Supplementary unit: {result.supplementary_unit} "
            f"({result.supplementary_unit_description})"
        )

    # ------------------------------------------------------------------
    # 2. Reusing one classifier (and one LangChain chat model) for many items.
    # ------------------------------------------------------------------
    llm = get_chat_model()
    classifier = HSCodeClassifier(llm=llm)

    products = [
        "Cotton T-shirt, white, men's",
        "Lithium-ion rechargeable battery, 18650, 3.7V",
        "Stainless steel screws M6 x 30mm zinc plated",
        "Glass jar with metal lid, 500ml, for jam packaging",
        "Wireless mouse, optical, USB receiver",
    ]
    for product in products:
        r = classifier.classify(product)
        marker = "✓" if r.validated else "✗"
        print(f"  {marker} {r.hs_code}  ({r.confidence:.0%})  {product[:50]}")


if __name__ == "__main__":
    main()
