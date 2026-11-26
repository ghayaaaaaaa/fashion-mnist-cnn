"""Command-line entry point for training and evaluating a CNN."""

import argparse
import json
from pathlib import Path
from typing import Sequence

from tensorflow import keras

from .data import load_fashion_mnist
from .evaluation import evaluate_classifier
from .model import MODEL_CONFIGS, build_classifier


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a configurable CNN on the Fashion-MNIST benchmark."
    )
    parser.add_argument(
        "--variant",
        choices=tuple(MODEL_CONFIGS),
        default="balanced",
        help="CNN capacity to use.",
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--validation-split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/fashion_mnist"),
    )
    return parser


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.epochs < 1:
        parser.error("--epochs must be at least 1.")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1.")
    if args.patience < 1:
        parser.error("--patience must be at least 1.")
    if not 0.05 <= args.validation_split <= 0.5:
        parser.error("--validation-split must be between 0.05 and 0.5.")

    keras.utils.set_random_seed(args.seed)
    data = load_fashion_mnist()
    model = build_classifier(args.variant)

    history = model.fit(
        data.train_images,
        data.train_labels,
        validation_split=args.validation_split,
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=args.patience,
                restore_best_weights=True,
            )
        ],
        verbose=2,
    )

    evaluation = evaluate_classifier(
        model,
        data.test_images,
        data.test_labels,
        batch_size=args.batch_size,
    )
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(output_dir / "fashion_mnist_cnn.keras")

    history_report = {
        name: [float(value) for value in values]
        for name, values in history.history.items()
    }
    _write_json(output_dir / "evaluation.json", evaluation)
    _write_json(
        output_dir / "training_history.json",
        {
            "variant": args.variant,
            "seed": args.seed,
            "epochs_requested": args.epochs,
            "batch_size": args.batch_size,
            "validation_split": args.validation_split,
            "history": history_report,
        },
    )

    print(
        json.dumps(
            {
                "output_dir": str(output_dir.resolve()),
                "test_accuracy": evaluation["test_accuracy"],
                "test_loss": evaluation["test_loss"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
