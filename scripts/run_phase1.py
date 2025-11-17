"""Command-line entry point for Phase 1 verification and preprocessing config."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tcm_pipeline import DataLoader, Preprocessor
from tcm_pipeline.reporting import ReportWriter


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run sampling-frequency checks and data-quality analysis.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("Data"),
        help="Directory that contains the vibration CSV files.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("reports") / "phase1",
        help="Destination directory for JSON/Markdown reports.",
    )
    return parser


def _aggregate_metrics(metrics: List[Dict[str, float]]) -> Dict[str, float]:
    fs_mean = [item["fs_mean"] for item in metrics]
    fs_std = [item["fs_std"] for item in metrics]
    dt_mean = [item["dt_mean"] for item in metrics]
    return {
        "dataset_count": len(metrics),
        "avg_sampling_frequency_hz": float(np.mean(fs_mean)),
        "std_sampling_frequency_hz": float(np.mean(fs_std)),
        "avg_dt_seconds": float(np.mean(dt_mean)),
    }


def _summarize_quality(quality_report: Dict[str, Dict]) -> Dict[str, List[str]]:
    flags: Dict[str, List[str]] = {
        "missing_data": [],
        "outliers": [],
        "flatlines": [],
        "drift": [],
    }
    for file_name, payload in quality_report.items():
        for channel, stats in payload["channels"].items():
            if stats["missing_count"] > 0:
                flags["missing_data"].append(f"{file_name}::{channel}")
            if stats["outlier_count"] > 0:
                flags["outliers"].append(f"{file_name}::{channel}")
            if stats["flatline_segments"] > 0:
                flags["flatlines"].append(f"{file_name}::{channel}")
            if stats["drift_flag"]:
                flags["drift"].append(f"{file_name}::{channel}")
    return flags


def main() -> None:
    parser = _build_argument_parser()
    args = parser.parse_args()
    loader = DataLoader(data_dir=args.data_dir)
    preprocessor = Preprocessor()

    sampling_metrics: Dict[str, Dict[str, float]] = {}
    metric_list: List[Dict[str, float]] = []
    quality_metrics: Dict[str, Dict] = {}

    for dataset in loader.iter_datasets():
        metrics = preprocessor.compute_sampling_metrics(dataset.dataframe)
        metrics["label"] = dataset.label
        sampling_metrics[dataset.path.name] = metrics
        metric_list.append(metrics)

        quality = preprocessor.build_quality_report(dataset.dataframe)
        quality_metrics[dataset.path.name] = {
            "label": dataset.label,
            "channels": quality,
        }

    target_fs = preprocessor.derive_target_sampling_rate(metric_list)
    filter_upper = round(min(target_fs / 2 - 1, 500.0), 3)
    preprocess_config = preprocessor.build_config(
        target_sampling_rate=target_fs,
        window_duration_seconds=1.0,
        overlap_ratio=0.5,
        filter_cutoff_hz=[1.0, filter_upper],
    )

    aggregate = _aggregate_metrics(metric_list)
    quality_summary = _summarize_quality(quality_metrics)

    writer = ReportWriter(output_dir=args.report_dir)
    writer.write_json(
        {
            "aggregate": aggregate,
            "per_file": sampling_metrics,
            "recommended_target_sampling_rate_hz": target_fs,
        },
        "sampling_frequency_report.json",
    )
    writer.write_json(
        {
            "summary": quality_summary,
            "per_file": quality_metrics,
        },
        "data_quality_report.json",
    )
    writer.write_json(
        preprocess_config.as_dict(),
        "preprocessing_config.json",
    )
    writer.write_markdown(
        "\n".join(
            [
                "# Phase 1 Verification Output",
                "",
                f"- Target sampling rate: **{target_fs} Hz**",
                f"- Window size: **{preprocess_config.window_size_samples} samples**",
                f"- Window overlap: **{int(preprocess_config.window_overlap * 100)}%**",
                f"- STFT window: **{preprocess_config.stft_window} ({preprocess_config.stft_window_size} samples)**",
                "",
                "Refer to the JSON files in this directory for detailed per-file diagnostics.",
            ]
        ),
        "README.md",
    )


if __name__ == "__main__":
    main()
