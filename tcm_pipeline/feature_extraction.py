"""Feature extraction utilities for vibration signals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd


@dataclass
class FeatureSet:
    """Simple container for combined time and frequency features."""

    features: pd.DataFrame
    metadata: Dict[str, str]


class FeatureExtractor:
    """Extracts statistical, temporal, and spectral features from signal windows."""

    def __init__(self, fft_size: int = 1024) -> None:
        self.fft_size = fft_size

    def _time_domain_features(self, window: np.ndarray) -> Dict[str, float]:
        """Return basic statistics for a single window."""
        eps = 1e-12
        rms = float(np.sqrt(np.mean(np.square(window))))
        features = {
            "mean": float(np.mean(window)),
            "std": float(np.std(window, ddof=0)),
            "rms": rms,
            "mad": float(np.mean(np.abs(window - np.mean(window)))),
            "skewness": float(pd.Series(window).skew()),
            "kurtosis": float(pd.Series(window).kurtosis()),
            "crest_factor": float(np.max(np.abs(window)) / (rms + eps)),
        }
        return features

    def _frequency_domain_features(
        self,
        window: np.ndarray,
        sampling_rate: float,
    ) -> Dict[str, float]:
        """Return a handful of FFT-based statistics for the window."""
        fft_length = len(window)
        if self.fft_size:
            fft_length = min(len(window), self.fft_size)
        window_segment = window[:fft_length]
        tapered = window_segment * np.hanning(len(window_segment))
        fft_vals = np.fft.rfft(tapered, n=len(window_segment))
        freqs = np.fft.rfftfreq(len(window_segment), d=1.0 / sampling_rate)
        magnitude = np.abs(fft_vals)
        energy = np.sum(np.square(magnitude))
        peak_idx = int(np.argmax(magnitude))
        centroid = float(np.sum(freqs * magnitude) / np.sum(magnitude))
        features = {
            "spectral_energy": float(energy),
            "peak_frequency": float(freqs[peak_idx]),
            "spectral_centroid": centroid,
            "band_power_low": float(np.trapz(magnitude[(freqs >= 0) & (freqs < sampling_rate / 6)], dx=1)),
            "band_power_mid": float(
                np.trapz(
                    magnitude[(freqs >= sampling_rate / 6) & (freqs < sampling_rate / 3)],
                    dx=1,
                )
            ),
            "band_power_high": float(
                np.trapz(
                    magnitude[(freqs >= sampling_rate / 3)],
                    dx=1,
                )
            ),
        }
        return features

    def describe_window(
        self,
        window: np.ndarray,
        sampling_rate: float,
        prefix: str,
    ) -> Dict[str, float]:
        """Combine time and frequency domain features for one channel/window."""
        features = {}
        features.update({f"{prefix}_time_{k}": v for k, v in self._time_domain_features(window).items()})
        features.update({f"{prefix}_freq_{k}": v for k, v in self._frequency_domain_features(window, sampling_rate).items()})
        return features
