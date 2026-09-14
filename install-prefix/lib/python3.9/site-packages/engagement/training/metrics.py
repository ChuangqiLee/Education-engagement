"""Classification metrics shared by all backbones."""

from typing import Dict, Sequence

import numpy as np


def classification_metrics(targets: Sequence[int], predictions: Sequence[int], num_classes: int) -> Dict[str, object]:
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

    precision, recall, f1, _ = precision_recall_fscore_support(
        targets, predictions, labels=list(range(num_classes)), average="macro", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "confusion_matrix": confusion_matrix(targets, predictions, labels=list(range(num_classes))).tolist(),
    }

