"""Hyperparameter tuning for DL models using Optuna + GroupKFold."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
import optuna
import tensorflow as tf
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

tf.get_logger().setLevel("ERROR")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune DL models via Optuna.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("artifacts") / "phase3" / "phase3_datasets.npz",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports") / "phase3" / "dl_tuning",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--trials", type=int, default=15)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_dataset(npz_path: Path) -> Dict[str, np.ndarray]:
    if not npz_path.exists():
        raise FileNotFoundError(f"Dataset not found: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def split_train_val(train_indices: np.ndarray, groups: np.ndarray, val_size: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    dummy = np.zeros(len(train_indices))
    rel_train_idx, rel_val_idx = next(splitter.split(dummy, groups=groups[train_indices]))
    return train_indices[rel_train_idx], train_indices[rel_val_idx]


def build_cnn1d(input_shape, num_classes, params):
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv1D(params["filters1"], params["kernel_size"], padding="same", activation="relu")(inputs)
    x = tf.keras.layers.Conv1D(params["filters2"], params["kernel_size"], padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling1D()(x)
    x = tf.keras.layers.Conv1D(params["filters3"], params["kernel_size"], padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dropout(params["dropout"])(x)
    x = tf.keras.layers.Dense(params["dense_units"], activation="relu")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs)


def build_lstm(input_shape, num_classes, params):
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(params["units1"], return_sequences=True))(inputs)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(params["units2"]))(x)
    x = tf.keras.layers.Dropout(params["dropout"])(x)
    x = tf.keras.layers.Dense(params["dense_units"], activation="relu")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    return tf.keras.Model(inputs, outputs)


def build_hybrid(time_shape, spec_shape, num_classes, params):
    time_input = tf.keras.layers.Input(shape=time_shape)
    x1 = tf.keras.layers.Conv1D(params["time_filters"], params["time_kernel"], padding="same", activation="relu")(time_input)
    x1 = tf.keras.layers.GlobalAveragePooling1D()(x1)

    spec_input = tf.keras.layers.Input(shape=spec_shape)
    x2 = tf.keras.layers.Conv2D(params["spec_filters"], (3, 3), padding="same", activation="relu")(spec_input)
    x2 = tf.keras.layers.GlobalAveragePooling2D()(x2)

    merged = tf.keras.layers.Concatenate()([x1, x2])
    merged = tf.keras.layers.Dense(params["dense_units"], activation="relu")(merged)
    merged = tf.keras.layers.Dropout(params["dropout"])(merged)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(merged)
    return tf.keras.Model([time_input, spec_input], outputs)


def compile_model(model: tf.keras.Model, lr: float) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )


def objective_factory(
    model_type: str,
    X_time: np.ndarray,
    X_spec: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    args: argparse.Namespace,
):
    input_time_shape = X_time.shape[1:]
    input_spec_shape = X_spec.shape[1:]
    num_classes = len(np.unique(y))
    gkf = GroupKFold(n_splits=args.folds)

    def objective(trial: optuna.Trial) -> float:
        tf.keras.backend.clear_session()
        if model_type == "cnn1d":
            params = {
                "filters1": trial.suggest_categorical("filters1", [32, 48, 64, 96]),
                "filters2": trial.suggest_categorical("filters2", [32, 48, 64, 96, 128]),
                "filters3": trial.suggest_categorical("filters3", [32, 64, 96, 128]),
                "kernel_size": trial.suggest_categorical("kernel_size", [3, 5]),
                "dropout": trial.suggest_float("dropout", 0.1, 0.5),
                "dense_units": trial.suggest_categorical("dense_units", [64, 96, 128, 192]),
            }
            build_fn = lambda: build_cnn1d(input_time_shape, num_classes, params)
        elif model_type == "lstm":
            params = {
                "units1": trial.suggest_categorical("units1", [32, 48, 64, 96]),
                "units2": trial.suggest_categorical("units2", [16, 32, 48, 64]),
                "dropout": trial.suggest_float("dropout", 0.1, 0.5),
                "dense_units": trial.suggest_categorical("dense_units", [64, 96, 128, 192]),
            }
            build_fn = lambda: build_lstm(input_time_shape, num_classes, params)
        else:
            params = {
                "time_filters": trial.suggest_categorical("time_filters", [32, 48, 64, 96]),
                "time_kernel": trial.suggest_categorical("time_kernel", [3, 5]),
                "spec_filters": trial.suggest_categorical("spec_filters", [16, 24, 32, 48]),
                "dense_units": trial.suggest_categorical("dense_units", [64, 96, 128, 192]),
                "dropout": trial.suggest_float("dropout", 0.1, 0.5),
            }
            build_fn = lambda: build_hybrid(input_time_shape, input_spec_shape, num_classes, params)

        lr = trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True)
        accuracies: List[float] = []
        for fold_idx, (train_idx_base, test_idx) in enumerate(gkf.split(X_time, y, groups)):
            train_idx, val_idx = split_train_val(train_idx_base, groups, args.val_size, args.seed + fold_idx)
            model = build_fn()
            compile_model(model, lr)
            callbacks = [
                tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
            ]

            if model_type == "cnn2d":
                train_inputs = X_spec[train_idx]
                val_inputs = X_spec[val_idx]
                test_inputs = X_spec[test_idx]
            elif model_type == "hybrid":
                train_inputs = (X_time[train_idx], X_spec[train_idx])
                val_inputs = (X_time[val_idx], X_spec[val_idx])
                test_inputs = (X_time[test_idx], X_spec[test_idx])
            else:
                train_inputs = X_time[train_idx]
                val_inputs = X_time[val_idx]
                test_inputs = X_time[test_idx]

            model.fit(
                train_inputs,
                y[train_idx],
                validation_data=(val_inputs, y[val_idx]),
                epochs=args.epochs,
                batch_size=args.batch_size,
                callbacks=callbacks,
                verbose=0,
            )
            _, test_acc = model.evaluate(test_inputs, y[test_idx], verbose=0)
            accuracies.append(test_acc)
            tf.keras.backend.clear_session()

        return float(np.mean(accuracies))

    return objective


def main() -> None:
    args = parse_args()
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    dataset = load_dataset(args.dataset)
    X_time = dataset["X_time"]
    X_spec = dataset["X_spec"]
    y = dataset["y"]
    groups = dataset["groups"]

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for model_type in ["cnn1d", "lstm", "hybrid"]:
        print(f"\n=== Tuning {model_type.upper()} ===")
        study = optuna.create_study(direction="maximize", study_name=model_type)
        objective = objective_factory(model_type, X_time, X_spec, y, groups, args)
        study.optimize(objective, n_trials=args.trials, show_progress_bar=True)
        best_trial = study.best_trial
        payload = {
            "best_value": best_trial.value,
            "best_params": best_trial.params,
            "trials": [
                {"number": t.number, "value": t.value, "params": t.params}
                for t in study.trials
            ],
        }
        results[model_type] = payload
        (output_dir / f"{model_type}_study.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    (output_dir / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("Tuning complete.")


if __name__ == "__main__":
    main()
