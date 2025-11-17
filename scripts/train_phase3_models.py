"""Train and evaluate Phase 3 deep learning models with GroupKFold."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

tf.get_logger().setLevel("ERROR")
MODEL_NAMES = ["cnn1d", "lstm", "cnn2d", "hybrid"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Phase 3 DL models with GroupKFold.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("artifacts") / "phase3" / "phase3_datasets.npz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports") / "phase3" / "dl",
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--val-size", type=float, default=0.1, help="Validation fraction within each training fold.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_dataset(npz_path: Path) -> Dict[str, np.ndarray]:
    if not npz_path.exists():
        raise FileNotFoundError(f"Dataset not found: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def build_cnn1d(input_shape: Tuple[int, int], num_classes: int) -> tf.keras.Model:
    inputs = tf.keras.layers.Input(shape=input_shape, name="cnn1d_input")
    x = tf.keras.layers.Conv1D(64, 3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.Conv1D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D()(x)
    x = tf.keras.layers.Conv1D(128, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs, name="cnn1d")


def build_lstm(input_shape: Tuple[int, int], num_classes: int) -> tf.keras.Model:
    inputs = tf.keras.layers.Input(shape=input_shape, name="lstm_input")
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True))(inputs)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32))(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs, name="lstm")


def build_cnn2d(input_shape: Tuple[int, int, int], num_classes: int) -> tf.keras.Model:
    inputs = tf.keras.layers.Input(shape=input_shape, name="cnn2d_input")
    x = tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs, name="cnn2d")


def build_hybrid(
    time_shape: Tuple[int, int],
    spec_shape: Tuple[int, int, int],
    num_classes: int,
) -> tf.keras.Model:
    time_input = tf.keras.layers.Input(shape=time_shape, name="time_input")
    x1 = tf.keras.layers.Conv1D(64, 3, padding="same", activation="relu")(time_input)
    x1 = tf.keras.layers.GlobalAveragePooling1D()(x1)

    spec_input = tf.keras.layers.Input(shape=spec_shape, name="spec_input")
    x2 = tf.keras.layers.Conv2D(32, (3, 3), padding="same", activation="relu")(spec_input)
    x2 = tf.keras.layers.GlobalAveragePooling2D()(x2)

    merged = tf.keras.layers.Concatenate()([x1, x2])
    merged = tf.keras.layers.Dense(128, activation="relu")(merged)
    merged = tf.keras.layers.Dropout(0.3)(merged)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(merged)
    return tf.keras.Model([time_input, spec_input], outputs, name="hybrid")


def compile_model(model: tf.keras.Model) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )


def capture_summary(model: tf.keras.Model, path: Path) -> None:
    lines: List[str] = []

    def _capture(line: str) -> None:
        lines.append(line)

    model.summary(print_fn=_capture)
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_history(history: tf.keras.callbacks.History, title: str, output_path: Path) -> None:
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="train_loss")
    plt.plot(history.history["val_loss"], label="val_loss")
    plt.legend()
    plt.title(f"{title} Loss")
    plt.subplot(1, 2, 2)
    plt.plot(history.history["accuracy"], label="train_acc")
    plt.plot(history.history["val_accuracy"], label="val_acc")
    plt.legend()
    plt.title(f"{title} Accuracy")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_confusion(y_true, y_pred, labels, title, output_path: Path) -> None:
    matrix = confusion_matrix(y_true, y_pred, normalize="true")
    plt.figure(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, cmap="Blues", xticklabels=labels, yticklabels=labels, fmt=".2f")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def prediction_vs_truth_plot(y_true, y_pred, labels, output_path: Path, window: int = 100) -> None:
    if len(y_true) < window:
        window = len(y_true)
    plt.figure(figsize=(12, 4))
    plt.plot(range(window), y_true[:window], label="True", marker="o")
    plt.plot(range(window), y_pred[:window], label="Predicted", marker="x")
    plt.yticks([0, 1, 2], labels)
    plt.xlabel("Sample")
    plt.ylabel("Class")
    plt.legend()
    plt.title("Prediction vs Truth (Slice)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def residual_histogram(y_true, probas, output_path: Path) -> None:
    true_probs = probas[np.arange(len(y_true)), y_true]
    plt.figure(figsize=(6, 4))
    sns.histplot(true_probs, bins=20, kde=True)
    plt.title("Distribution of True-Class Probabilities")
    plt.xlabel("Probability")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def split_train_val(train_indices: np.ndarray, groups: np.ndarray, val_size: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    dummy = np.zeros(len(train_indices))
    rel_train_idx, rel_val_idx = next(splitter.split(dummy, groups=groups[train_indices]))
    return train_indices[rel_train_idx], train_indices[rel_val_idx]


def main() -> None:
    args = parse_args()
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    data = load_dataset(args.dataset)
    X_time = data["X_time"]
    X_spec = data["X_spec"]
    y = data["y"]
    groups = data["groups"]

    dataset_shapes = {
        "X_time": list(X_time.shape),
        "X_spec": list(X_spec.shape),
        "y": list(y.shape),
        "groups_unique": int(len(np.unique(groups))),
    }

    output_dir = ensure_dir(args.output_dir)
    (output_dir / "dataset_shapes.json").write_text(json.dumps(dataset_shapes, indent=2), encoding="utf-8")

    input_time_shape = X_time.shape[1:]
    input_spec_shape = X_spec.shape[1:]
    num_classes = len(np.unique(y))

    model_builders = {
        "cnn1d": lambda: build_cnn1d(input_time_shape, num_classes),
        "lstm": lambda: build_lstm(input_time_shape, num_classes),
        "cnn2d": lambda: build_cnn2d(input_spec_shape, num_classes),
        "hybrid": lambda: build_hybrid(input_time_shape, input_spec_shape, num_classes),
    }

    gkf = GroupKFold(n_splits=args.folds)

    overall_results: List[Dict[str, float]] = []
    for model_id in MODEL_NAMES:
        print(f"\n=== Training {model_id.upper()} with GroupKFold ===")
        model_dir = ensure_dir(output_dir / model_id)
        fold_metrics: List[Dict[str, float]] = []
        all_true: List[np.ndarray] = []
        all_pred: List[np.ndarray] = []
        all_proba: List[np.ndarray] = []

        for fold_idx, (train_idx_base, test_idx) in enumerate(gkf.split(X_time, y, groups)):
            inner_train_idx, val_idx = split_train_val(train_idx_base, groups, args.val_size, args.seed + fold_idx)
            model = model_builders[model_id]()
            compile_model(model)
            fold_dir = ensure_dir(model_dir / f"fold_{fold_idx + 1}")
            capture_summary(model, fold_dir / "model_summary.txt")

            train_inputs = {
                "cnn1d": X_time[inner_train_idx],
                "lstm": X_time[inner_train_idx],
                "cnn2d": X_spec[inner_train_idx],
                "hybrid": (X_time[inner_train_idx], X_spec[inner_train_idx]),
            }[model_id]
            val_inputs = {
                "cnn1d": X_time[val_idx],
                "lstm": X_time[val_idx],
                "cnn2d": X_spec[val_idx],
                "hybrid": (X_time[val_idx], X_spec[val_idx]),
            }[model_id]
            test_inputs = {
                "cnn1d": X_time[test_idx],
                "lstm": X_time[test_idx],
                "cnn2d": X_spec[test_idx],
                "hybrid": (X_time[test_idx], X_spec[test_idx]),
            }[model_id]

            callbacks = [
                tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
            ]

            history = model.fit(
                train_inputs,
                y[inner_train_idx],
                validation_data=(val_inputs, y[val_idx]),
                epochs=args.epochs,
                batch_size=args.batch_size,
                callbacks=callbacks,
                verbose=0,
            )
            plot_history(history, f"{model_id.upper()} Fold {fold_idx + 1}", fold_dir / "training_curves.png")

            test_loss, test_acc = model.evaluate(test_inputs, y[test_idx], verbose=0)
            probas = model.predict(test_inputs, verbose=0)
            preds = np.argmax(probas, axis=1)

            fold_metrics.append(
                {
                    "fold": fold_idx + 1,
                    "test_loss": float(test_loss),
                    "test_accuracy": float(test_acc),
                }
            )

            all_true.append(y[test_idx])
            all_pred.append(preds)
            all_proba.append(probas)

        y_true = np.concatenate(all_true)
        y_pred = np.concatenate(all_pred)
        y_proba = np.concatenate(all_proba)
        report = classification_report(y_true, y_pred, output_dict=True)
        mean_loss = float(np.mean([m["test_loss"] for m in fold_metrics]))
        mean_acc = float(np.mean([m["test_accuracy"] for m in fold_metrics]))

        metrics_payload = {
            "folds": fold_metrics,
            "aggregate_accuracy": mean_acc,
            "aggregate_loss": mean_loss,
            "classification_report": report,
        }
        (model_dir / "metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
        plot_confusion(
            y_true,
            y_pred,
            labels=["Good (0)", "Moderate (1)", "Bad (2)"],
            title=f"{model_id.upper()} Confusion Matrix",
            output_path=model_dir / "confusion_matrix.png",
        )
        prediction_vs_truth_plot(
            y_true,
            y_pred,
            ["Good", "Moderate", "Bad"],
            model_dir / "prediction_vs_truth.png",
        )
        residual_histogram(y_true, y_proba, model_dir / "residual_histogram.png")
        np.save(model_dir / "y_true.npy", y_true)
        np.save(model_dir / "y_pred.npy", y_pred)
        np.save(model_dir / "y_proba.npy", y_proba)

        overall_results.append({"model": model_id, "accuracy": mean_acc, "loss": mean_loss})

    (output_dir / "model_comparison.json").write_text(json.dumps(overall_results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
