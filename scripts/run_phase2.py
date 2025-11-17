"""Phase 2 pipeline: feature extraction + unsupervised exploration with plots."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tcm_pipeline import (
    DataLoader,
    FeatureExtractor,
    Preprocessor,
    PreprocessingConfig,
    UnsupervisedModel,
)
from tcm_pipeline.data_loader import CORE_CHANNELS
from tcm_pipeline.reporting import ReportWriter


def _parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Phase 2 feature extraction and unsupervised exploration.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("Data"),
        help="Directory containing the vibration CSV files.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("reports") / "phase1" / "preprocessing_config.json",
        help="Path to preprocessing_config.json (auto-derived if missing).",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("reports") / "phase2",
        help="Directory for Phase 2 outputs (features, plots, metrics).",
    )
    return parser


def _load_or_create_config(
    config_path: Path,
    loader: DataLoader,
    preprocessor: Preprocessor,
) -> PreprocessingConfig:
    if config_path.exists():
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        return PreprocessingConfig(**payload)

    metrics: List[Dict[str, float]] = []
    for dataset in loader.iter_datasets():
        metrics.append(preprocessor.compute_sampling_metrics(dataset.dataframe))
    target_fs = preprocessor.derive_target_sampling_rate(metrics)
    return preprocessor.build_config(
        target_sampling_rate=target_fs,
        window_duration_seconds=1.0,
        overlap_ratio=0.5,
        filter_cutoff_hz=[1.0, max(1.0, round(target_fs / 2 - 1, 3))],
    )


def _standardize_window(window: np.ndarray) -> np.ndarray:
    mean = window.mean(axis=0, keepdims=True)
    std = window.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    return (window - mean) / std


def _channel_slug(channel: str) -> str:
    return (
        channel.lower()
        .replace("(°/s)", "dps")
        .replace("(g)", "g")
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("/", "_")
    )


def _plot_embedding(
    embedding: np.ndarray,
    labels: List[str],
    title: str,
    palette: Dict[str, str],
    output_path: Path,
) -> None:
    plt.figure(figsize=(8, 6))
    label_series = pd.Series(labels)
    colors = label_series.map(palette)
    plt.scatter(embedding[:, 0], embedding[:, 1], c=colors, s=18, alpha=0.85, edgecolors="none")
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", label=lbl, markerfacecolor=col, markersize=8)
        for lbl, col in palette.items()
    ]
    plt.legend(handles=legend_handles, loc="best")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def _reset_phase2_outputs(report_dir: Path) -> None:
    """Remove stale outputs so reruns always reflect the latest config."""
    if not report_dir.exists():
        return
    for artifact in [
        report_dir / "window_features.csv",
        report_dir / "unsupervised_summary.json",
        report_dir / "README.md",
    ]:
        if artifact.exists():
            artifact.unlink()
    figures_dir = report_dir / "figures"
    if figures_dir.exists():
        for image in figures_dir.glob("*.png"):
            image.unlink()


def main() -> None:
    parser = _parse_args()
    args = parser.parse_args()

    loader = DataLoader(data_dir=args.data_dir)
    preprocessor = Preprocessor()
    config = _load_or_create_config(args.config, loader, preprocessor)
    extractor = FeatureExtractor(fft_size=config.stft_window_size)
    unsupervised = UnsupervisedModel()
    _reset_phase2_outputs(args.report_dir)

    feature_rows: List[Dict[str, float]] = []
    labels: List[str] = []
    files: List[str] = []
    window_indices: List[int] = []

    for dataset in loader.iter_datasets():
        resampled_df = preprocessor.resample_dataframe(
            dataset.dataframe,
            target_fs=config.target_sampling_rate_hz,
            channels=CORE_CHANNELS,
        )
        signal = resampled_df[CORE_CHANNELS].to_numpy()
        windows, bounds = preprocessor.window_signal(
            signal,
            window_size=config.window_size_samples,
            overlap=config.window_overlap,
        )
        if not len(bounds):
            continue
        for idx, window in enumerate(windows):
            normalized = _standardize_window(window)
            features: Dict[str, float] = {}
            for channel_idx, channel in enumerate(CORE_CHANNELS):
                slug = _channel_slug(channel)
                channel_features = extractor.describe_window(
                    normalized[:, channel_idx],
                    sampling_rate=config.target_sampling_rate_hz,
                    prefix=slug,
                )
                features.update(channel_features)
            features["file"] = dataset.path.name
            features["label"] = dataset.label
            features["window_index"] = idx
            feature_rows.append(features)
            labels.append(dataset.label)
            files.append(dataset.path.name)
            window_indices.append(idx)

    if not feature_rows:
        raise RuntimeError("No windows were generated. Check window size vs. recording length.")

    feature_df = pd.DataFrame(feature_rows)
    meta_columns = ["file", "label", "window_index"]
    feature_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    feature_df.dropna(axis=1, how="all", inplace=True)
    feature_df.dropna(axis=0, inplace=True)

    meta = feature_df[meta_columns].reset_index(drop=True)
    X = feature_df.drop(columns=meta_columns)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    if len(meta) == 3310:
        raise RuntimeError(
            "Window count unexpectedly remained at 3310. "
            "Ensure the preprocessing configuration change was applied."
        )

    umap_result = unsupervised.analyze(pd.DataFrame(X_scaled, columns=X.columns), method="umap")
    pca_embedding = unsupervised.run_pca(pd.DataFrame(X_scaled, columns=X.columns))

    aria = adjusted_rand_score(meta["label"], umap_result.clusters)
    nmi = normalized_mutual_info_score(meta["label"], umap_result.clusters)
    silhouette = (
        float(silhouette_score(X_scaled, umap_result.clusters))
        if len(set(umap_result.clusters)) > 1
        else float("nan")
    )

    palette = {
        "good": "#1f77b4",
        "moderate": "#ff7f0e",
        "bad": "#d62728",
    }
    report_dir = args.report_dir
    figures_dir = report_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    _plot_embedding(
        umap_result.embedding,
        meta["label"].tolist(),
        "UMAP Embedding Colored by Label",
        palette=palette,
        output_path=figures_dir / "umap_by_label.png",
    )
    unique_clusters = sorted(set(int(c) for c in umap_result.clusters))
    palette_values = sns.color_palette("husl", len(unique_clusters))
    cluster_palette = {str(cluster): palette_values[idx] for idx, cluster in enumerate(unique_clusters)}
    _plot_embedding(
        umap_result.embedding,
        [str(c) for c in umap_result.clusters],
        "UMAP Embedding Colored by KMeans Clusters",
        palette=cluster_palette,
        output_path=figures_dir / "umap_by_cluster.png",
    )
    _plot_embedding(
        pca_embedding,
        meta["label"].tolist(),
        "PCA Embedding Colored by Label",
        palette=palette,
        output_path=figures_dir / "pca_by_label.png",
    )

    writer = ReportWriter(output_dir=report_dir)
    writer.write_json(
        {
            "windows": len(meta),
            "features": list(X.columns),
            "cluster_metrics": {
                "adjusted_rand_index": aria,
                "normalized_mutual_information": nmi,
                "silhouette": silhouette,
            },
        },
        "unsupervised_summary.json",
    )
    feature_df.to_csv(report_dir / "window_features.csv", index=False)
    writer.write_markdown(
        "\n".join(
            [
                "# Phase 2 Unsupervised Insights",
                "",
                f"- Windows analyzed: **{len(meta)}**",
                f"- Features per window: **{X.shape[1]}**",
                f"- Adjusted Rand Index (labels vs. clusters): **{aria:.3f}**",
                f"- Normalized Mutual Information: **{nmi:.3f}**",
                f"- Silhouette Score: **{silhouette:.3f}**",
                "",
                "See the figures in this directory for the UMAP/PCA scatter plots. "
                "Use `window_features.csv` for downstream classical/NN experiments.",
            ]
        ),
        "README.md",
    )


if __name__ == "__main__":
    main()
