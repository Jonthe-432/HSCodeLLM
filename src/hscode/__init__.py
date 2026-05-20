"""
HSCode — Model-agnostic EU HS / Combined Nomenclature classifier.

Public API:

    from hscode import classify, HSCodeClassifier, ClassificationResult

    result = classify("Wireless bluetooth headphones",
                      provider="openai", model="gpt-5.4-nano")
    print(result.hs_code, result.confidence)
"""

from hscode.api import classify
from hscode.classifier import HSCodeClassifier
from hscode.models import ClassificationResult

__all__ = [
    "classify",
    "HSCodeClassifier",
    "ClassificationResult",
]

__version__ = "0.1.0"
