"""Evaluation helpers that produce machine-readable reports."""

import numpy as np
from tensorflow import keras

from .model import CLASS_NAMES


def evaluate_classifier(
    model: keras.Model,
    test_images: np.ndarray,
    test_labels: np.ndarray,
    batch_size: int,
) -> dict[str, object]:
    """Evaluate the test split and compute per-class metrics."""
    scores = model.evaluate(
        test_images,
        test_labels,
        batch_size=batch_size,
        verbose=0,
        return_dict=True,
    )
    probabilities = model.predict(test_images, batch_size=batch_size, verbose=0)
    predictions = np.argmax(probabilities, axis=1)

    confusion = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    np.add.at(confusion, (test_labels, predictions), 1)

    per_class: dict[str, dict[str, float | int]] = {}
    for class_index, class_name in enumerate(CLASS_NAMES):
        true_positive = int(confusion[class_index, class_index])
        support = int(confusion[class_index, :].sum())
        predicted_count = int(confusion[:, class_index].sum())
        precision = true_positive / predicted_count if predicted_count else 0.0
        recall = true_positive / support if support else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        per_class[class_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }

    return {
        "test_examples": int(test_labels.shape[0]),
        "test_loss": float(scores["loss"]),
        "test_accuracy": float(scores["accuracy"]),
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
    }
