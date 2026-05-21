from .base import BaseClassifier, ClassificationResult, ImageClassificationResult, FailureType, FeatureVector
from .features import extract_features
from .severity import area_to_severity, worst_severity, load_thresholds
from .rules import classify_features
from .classifier import RuleBasedClassifier
from .pipeline import classify_all, ClassificationSummary
from .store import ensure_table, get_classification_runs, upsert_classification_run

__all__ = [
    "BaseClassifier", "ClassificationResult", "ImageClassificationResult",
    "FailureType", "FeatureVector",
    "extract_features",
    "area_to_severity", "worst_severity", "load_thresholds",
    "classify_features",
    "RuleBasedClassifier",
    "classify_all", "ClassificationSummary",
    "ensure_table", "get_classification_runs", "upsert_classification_run",
]
