"""Preprocessing utilities for vibration data quality assessment and signal prep."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import interpolate, signal


@dataclass
class PreprocessingConfig:
    """Encapsulates the preprocessing parameters used across the pipeline."""

    target_sampling_rate_hz: float
    window_size_samples: int
    window_overlap: float
    filtering_method: str
    filter_order: int
    filter_cutoff_hz: Optional[List[float]]
    normalization: str
    detrending: str
    stft_window: str
    stft_window_size: int
    stft_overlap: float

    def as_dict(self) -> Dict[str, float]:
        """Return a JSON-serializable dictionary representation."""
        return asdict(self)


class Preprocessor:
    """Provides data verification, resampling, and preprocessing helpers."""

    def __init__(
        self,
        mad_threshold: float = 3.5,
        flatline_min_samples: int = 60,
        flatline_tolerance: float = 1e-4,
        drift_slope_threshold: float = 5e-4,
    ) -> None:
        self.mad_threshold = mad_threshold
        self.flatline_min_samples = flatline_min_samples
        self.flatline_tolerance = flatline_tolerance
        self.drift_slope_threshold = drift_slope_threshold

    @staticmethod
    def _parse_chip_time(series: pd.Series) -> pd.Series:
        cleaned = series.astype(str).str.strip()
        datetimes = pd.to_datetime(cleaned, errors="coerce")
        if datetimes.isna().all():
            raise ValueError("Unable to parse any timestamps from 'Chip Time()' column.")
        return datetimes

    def compute_sampling_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Compute dt and sampling-rate statistics for a dataframe."""
        if "Chip Time()" not in df.columns:
            raise ValueError("Expected 'Chip Time()' column in dataframe.")
        timestamps = self._parse_chip_time(df["Chip Time()"])
        dt = timestamps.diff().dt.total_seconds().dropna()
        dt = dt[dt > 0]
        if dt.empty:
            raise ValueError("Unable to compute sampling deltas; inspect timestamp data.")
        fs = 1.0 / dt
        metrics = {
            "samples": int(len(df)),
            "duration_seconds": float((timestamps.iloc[-1] - timestamps.iloc[0]).total_seconds()),
            "dt_mean": float(dt.mean()),
            "dt_std": float(dt.std(ddof=0)),
            "fs_mean": float(fs.mean()),
            "fs_std": float(fs.std(ddof=0)),
            "fs_median": float(fs.median()),
        }
        return metrics

    def _mad_based_outliers(self, series: pd.Series) -> Dict[str, float]:
        values = series.dropna().to_numpy()
        if values.size == 0:
            return {"count": 0, "ratio": 0.0}
        median = np.median(values)
        mad = np.median(np.abs(values - median))
        if mad == 0:
            return {"count": 0, "ratio": 0.0}
        modified_z = 0.6745 * (values - median) / mad
        mask = np.abs(modified_z) > self.mad_threshold
        count = int(mask.sum())
        ratio = count / len(series)
        return {"count": count, "ratio": float(ratio)}

    def _flatline_stats(self, series: pd.Series) -> Dict[str, float]:
        values = series.ffill().bfill().to_numpy()
        if values.size <= 1:
            return {"segments": 0, "longest": 0}
        diffs = np.abs(np.diff(values))
        mask = diffs < self.flatline_tolerance
        segments = 0
        longest = 0
        current = 0
        for is_flat in mask:
            if is_flat:
                current += 1
            else:
                if current >= self.flatline_min_samples:
                    segments += 1
                    longest = max(longest, current + 1)
                current = 0
        if current >= self.flatline_min_samples:
            segments += 1
            longest = max(longest, current + 1)
        return {"segments": segments, "longest": int(longest)}

    def _drift_stat(self, series: pd.Series, timestamps: pd.Series) -> float:
        times = timestamps.astype("int64") / 1e9
        values = series.to_numpy()
        mask = np.isfinite(values)
        if mask.sum() < 2:
            return 0.0
        slope, _ = np.polyfit(times[mask], values[mask], deg=1)
        return float(slope)

    def build_quality_report(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Generate missing-value, outlier, flatline, and drift stats per channel."""
        timestamps = self._parse_chip_time(df["Chip Time()"])
        report: Dict[str, Dict[str, float]] = {}
        for column in [
            "Acceleration X(g)",
            "Acceleration Y(g)",
            "Acceleration Z(g)",
            "Angular velocity X(°/s)",
            "Angular velocity Y(°/s)",
            "Angular velocity Z(°/s)",
        ]:
            series = pd.to_numeric(df[column], errors="coerce")
            missing = int(series.isna().sum())
            outliers = self._mad_based_outliers(series)
            flatline = self._flatline_stats(series)
            drift = self._drift_stat(series, timestamps)
            report[column] = {
                "missing_count": missing,
                "missing_ratio": float(missing / len(series)),
                "outlier_count": outliers["count"],
                "outlier_ratio": outliers["ratio"],
                "flatline_segments": flatline["segments"],
                "flatline_longest_samples": flatline["longest"],
                "drift_slope_per_sec": drift,
                "drift_flag": abs(drift) > self.drift_slope_threshold,
            }
        return report

    def derive_target_sampling_rate(self, metrics: List[Dict[str, float]]) -> float:
        """Choose a common target sampling rate based on per-file stats."""
        freq_values = [item["fs_median"] for item in metrics if not np.isnan(item["fs_median"])]
        if not freq_values:
            raise ValueError("No sampling frequency values available for aggregation.")
        target = float(np.median(freq_values))
        return round(target, 3)

    def _design_bandpass(
        self,
        sampling_rate: float,
        cutoff_hz: List[float],
        order: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return IIR coefficients for a Butterworth band-pass filter."""
        nyquist = 0.5 * sampling_rate
        low = max(cutoff_hz[0] / nyquist, 1e-4)
        high = min(cutoff_hz[1] / nyquist, 0.999)
        if low >= high:
            raise ValueError("Invalid band-pass cutoff; ensure low < high < Nyquist.")
        return signal.butter(order, [low, high], btype="bandpass")

    def apply_filter(
        self,
        data: np.ndarray,
        sampling_rate: float,
        method: str,
        cutoff_hz: Optional[List[float]],
        order: int,
    ) -> np.ndarray:
        """Filter a multi-channel signal along the time axis."""
        if method.lower() != "bandpass":
            return data
        if not cutoff_hz:
            return data
        b, a = self._design_bandpass(sampling_rate, cutoff_hz, order)
        return signal.filtfilt(b, a, data, axis=0)

    def filter_dataframe(
        self,
        df: pd.DataFrame,
        sampling_rate: float,
        method: str,
        cutoff_hz: Optional[List[float]],
        order: int,
        channels: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Apply the configured filter to the desired sensor channels."""
        columns = channels or [
            "Acceleration X(g)",
            "Acceleration Y(g)",
            "Acceleration Z(g)",
            "Angular velocity X(°/s)",
            "Angular velocity Y(°/s)",
            "Angular velocity Z(°/s)",
        ]
        data = df[columns].to_numpy()
        filtered = self.apply_filter(
            data=data,
            sampling_rate=sampling_rate,
            method=method,
            cutoff_hz=cutoff_hz,
            order=order,
        )
        result = df.copy()
        for idx, column in enumerate(columns):
            result[column] = filtered[:, idx]
        return result

    def resample_channel(
        self,
        timestamps: pd.Series,
        series: pd.Series,
        target_fs: float,
    ) -> pd.Series:
        """Return an interpolated series at the requested sampling rate."""
        cleaned_time = timestamps - timestamps.iloc[0]
        seconds = cleaned_time.dt.total_seconds().to_numpy()
        values = pd.to_numeric(series, errors="coerce").to_numpy()
        mask = np.isfinite(seconds) & np.isfinite(values)
        if mask.sum() < 4:
            raise ValueError("Not enough valid samples for interpolation.")
        sec_masked = seconds[mask]
        val_masked = values[mask]
        order = np.argsort(sec_masked)
        sec_sorted = sec_masked[order]
        val_sorted = val_masked[order]
        unique_seconds, unique_indices = np.unique(sec_sorted, return_index=True)
        unique_values = val_sorted[unique_indices]
        if unique_seconds.size < 4:
            raise ValueError("Not enough unique timestamps for interpolation.")
        cs = interpolate.CubicSpline(unique_seconds, unique_values)
        new_times = np.arange(0, unique_seconds.max(), 1.0 / target_fs)
        resampled = cs(new_times)
        return pd.Series(resampled)

    def resample_dataframe(
        self,
        df: pd.DataFrame,
        target_fs: float,
        channels: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Resample all requested channels to a common sampling rate."""
        columns = channels or [
            "Acceleration X(g)",
            "Acceleration Y(g)",
            "Acceleration Z(g)",
            "Angular velocity X(°/s)",
            "Angular velocity Y(°/s)",
            "Angular velocity Z(°/s)",
        ]
        timestamps = self._parse_chip_time(df["Chip Time()"])
        resampled = {}
        for column in columns:
            resampled[column] = self.resample_channel(timestamps, df[column], target_fs)
        length = len(next(iter(resampled.values())))
        timeline = timestamps.iloc[0] + pd.to_timedelta(np.arange(length) / target_fs, unit="s")
        resampled_df = pd.DataFrame(resampled)
        resampled_df["Chip Time()"] = timeline
        return resampled_df

    def window_signal(
        self,
        data: np.ndarray,
        window_size: int,
        overlap: float,
    ) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """Return overlapping windows for a multichannel signal."""
        if not 0 <= overlap < 1:
            raise ValueError("overlap must be between 0 (inclusive) and 1 (exclusive).")
        step = max(1, int(window_size * (1 - overlap)))
        if data.shape[0] < window_size:
            return np.empty((0, window_size, data.shape[1])), []
        windows = []
        bounds: List[Tuple[int, int]] = []
        for start in range(0, data.shape[0] - window_size + 1, step):
            end = start + window_size
            windows.append(data[start:end])
            bounds.append((start, end))
        return np.stack(windows), bounds

    def build_config(
        self,
        target_sampling_rate: float,
        window_duration_seconds: float = 1.0,
        overlap_ratio: float = 0.5,
        filter_cutoff_hz: Optional[List[float]] = None,
    ) -> PreprocessingConfig:
        """Create a preprocessing configuration object using derived stats."""
        window_size = int(target_sampling_rate * window_duration_seconds)
        stft_window_size = min(window_size, 256)
        return PreprocessingConfig(
            target_sampling_rate_hz=target_sampling_rate,
            window_size_samples=window_size,
            window_overlap=overlap_ratio,
            filtering_method="bandpass" if filter_cutoff_hz else "none",
            filter_order=4,
            filter_cutoff_hz=filter_cutoff_hz,
            normalization="per_window_standardization",
            detrending="mean_subtraction",
            stft_window="hann",
            stft_window_size=stft_window_size,
            stft_overlap=0.75,
        )
