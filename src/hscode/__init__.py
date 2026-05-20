"""
HSCode — Model-agnostic EU HS / Combined Nomenclature classifier.

Built on LangChain. Public API:

    from hscode import classify, HSCodeClassifier, ClassificationResult

    result = classify("Wireless bluetooth headphones",
                      provider="openai", model="gpt-5.4-nano")
    print(result.hs_code, result.confidence)
"""

from hscode.api import classify
from hscode.classifier import HSCodeClassifier
from hscode.llm import get_chat_model, list_providers
from hscode.models import ClassificationResult

__all__ = [
    "classify",
    "HSCodeClassifier",
    "ClassificationResult",
    "get_chat_model",
    "list_providers",
]

__version__ = "0.2.0"
