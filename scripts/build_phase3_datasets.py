"""Generate windowed datasets with group metadata for Phase 3 models."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy import signal

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tcm_pipeline import DataLoader, Preprocessor, PreprocessingConfig
from tcm_pipeline.data_loader import CORE_CHANNELS


LABEL_MAP = {"good": 0, "moderate": 1, "bad": 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Phase 3 datasets (time windows + spectrograms) with filtering.",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("Data"))
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("reports") / "phase1" / "preprocessing_config.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts") / "phase3",
    )
    parser.add_argument("--binary", action="store_true", help="Merge Moderate and Bad into a single Faulty class.")
    parser.add_argument("--stft-size", type=int, default=None, help="Override STFT window size (e.g., 1024).")
    parser.add_argument("--stft-overlap", type=float, default=None, help="Override STFT overlap (e.g., 0.9).")
    return parser.parse_args()


def load_config(path: Path) -> PreprocessingConfig:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return PreprocessingConfig(**payload)


def compute_spectrogram(
    window: np.ndarray,
    sampling_rate: float,
    n_fft: int,
    overlap: float,
) -> np.ndarray:
    hop = max(1, int(n_fft * (1 - overlap)))
    noverlap = n_fft - hop
    
    # Pad window if it's shorter than n_fft
    if window.shape[0] < n_fft:
        pad_width = n_fft - window.shape[0]
        window = np.pad(window, ((0, pad_width), (0, 0)), mode='constant')

    channel_specs: List[np.ndarray] = []
    for channel_idx in range(window.shape[1]):
        _, _, Zxx = signal.stft(
            window[:, channel_idx],
            fs=sampling_rate,
            nperseg=n_fft,
            noverlap=noverlap,
            padded=True,  # Enable padding in STFT as well
            boundary=None,
        )
        magnitude = np.abs(Zxx)
        channel_specs.append(magnitude)
    averaged = np.stack(channel_specs, axis=0).mean(axis=0)
    return 20 * np.log10(np.maximum(averaged, 1e-12))


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    loader = DataLoader(args.data_dir)
    preprocessor = Preprocessor()

    X_time: List[np.ndarray] = []
    X_spec: List[np.ndarray] = []
    y: List[int] = []
    group_ids: List[int] = []
    file_names: List[str] = []
    window_offsets: List[int] = []
    group_lookup: Dict[str, int] = {}

    for dataset in loader.iter_datasets():
        df = preprocessor.resample_dataframe(
            dataset.dataframe,
            target_fs=config.target_sampling_rate_hz,
            channels=CORE_CHANNELS,
        )
        if config.filtering_method.lower() == "bandpass":
            df = preprocessor.filter_dataframe(
                df,
                sampling_rate=config.target_sampling_rate_hz,
                method=config.filtering_method,
                cutoff_hz=config.filter_cutoff_hz,
                order=config.filter_order,
                channels=CORE_CHANNELS,
            )
        data = df[CORE_CHANNELS].to_numpy()
        windows, bounds = preprocessor.window_signal(
            data,
            window_size=config.window_size_samples,
            overlap=config.window_overlap,
        )
        if windows.size == 0:
            continue
        group_idx = group_lookup.setdefault(dataset.path.name, len(group_lookup))
        
        # Label mapping logic
        raw_label_idx = LABEL_MAP.get(dataset.label, -1)
        if args.binary:
            # 0 (Good) -> 0
            # 1 (Moderate) -> 1
            # 2 (Bad) -> 1
            label_idx = 1 if raw_label_idx >= 1 else 0
        else:
            label_idx = raw_label_idx
            
        # Determine STFT params
        n_fft = args.stft_size if args.stft_size is not None else config.stft_window_size
        overlap = args.stft_overlap if args.stft_overlap is not None else config.stft_overlap

        for idx, window in enumerate(windows):
            spec = compute_spectrogram(
                window,
                sampling_rate=config.target_sampling_rate_hz,
                n_fft=n_fft,
                overlap=overlap,
            )
            X_time.append(window)
            X_spec.append(spec[..., np.newaxis])
            y.append(label_idx)
            group_ids.append(group_idx)
            file_names.append(dataset.path.name)
            window_offsets.append(bounds[idx][0])

    if not X_time:
        raise RuntimeError("No windows were generated; check the preprocessing config.")

    X_time_arr = np.stack(X_time)
    X_spec_arr = np.stack(X_spec)
    y_arr = np.array(y, dtype=np.int64)
    groups_arr = np.array(group_ids, dtype=np.int64)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "phase3_datasets.npz",
        X_time=X_time_arr,
        X_spec=X_spec_arr,
        y=y_arr,
        groups=groups_arr,
        file_names=np.array(file_names),
        window_offsets=np.array(window_offsets),
    )

    summary = {
        "samples": int(len(y_arr)),
        "time_series_shape": list(X_time_arr.shape),
        "spectrogram_shape": list(X_spec_arr.shape),
        "label_distribution": {
            str(label): int((y_arr == label).sum()) for label in sorted(np.unique(y_arr))
        },
        "group_counts": {
            name: int((groups_arr == idx).sum()) for name, idx in group_lookup.items()
        },
        "spectrogram_freq_bins": int(X_spec_arr.shape[1]),
        "spectrogram_time_bins": int(X_spec_arr.shape[2]),
    }
    (output_dir / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("Dataset creation complete.")
    print(f"Windows: {X_time_arr.shape}, Spectrograms: {X_spec_arr.shape}")
    print(f"Labels: {summary['label_distribution']}")
    print(f"Groups: {len(group_lookup)}")


if __name__ == "__main__":
    main()
