"""Utilities for loading vibration datasets from CSV files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

import pandas as pd


CORE_CHANNELS = [
    "Acceleration X(g)",
    "Acceleration Y(g)",
    "Acceleration Z(g)",
    "Angular velocity X(°/s)",
    "Angular velocity Y(°/s)",
    "Angular velocity Z(°/s)",
]

TIMESTAMP_COLUMN = "Chip Time()"


@dataclass(frozen=True)
class LoadedDataset:
    """Container holding the metadata and dataframe for a single CSV file."""

    path: Path
    label: str
    dataframe: pd.DataFrame


class DataLoader:
    """Loads vibration CSV files and performs basic validation."""

    def __init__(
        self,
        data_dir: Path,
        pattern: str = "*.csv",
        encoding: str = "utf-8-sig",
        core_channels: Optional[Iterable[str]] = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.pattern = pattern
        self.encoding = encoding
        self.core_channels = list(core_channels) if core_channels else CORE_CHANNELS
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

    def list_files(self) -> List[Path]:
        """Return all CSV files that match the configured pattern."""
        files = sorted(self.data_dir.glob(self.pattern))
        if not files:
            raise FileNotFoundError(
                f"No CSV files that match '{self.pattern}' were found in {self.data_dir}"
            )
        return files

    def _infer_label(self, path: Path) -> str:
        """Infer the class label from the filename."""
        name = path.name.lower()
        if "good" in name:
            return "good"
        if "moderate" in name:
            return "moderate"
        if "bad" in name:
            return "bad"
        return "unknown"

    def load_file(self, file_path: Path) -> pd.DataFrame:
        """Read a single CSV file and validate the expected columns."""
        try:
            df = pd.read_csv(
                file_path,
                encoding=self.encoding,
                index_col=False,
                skipinitialspace=True,
            )
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Unable to locate {file_path}") from exc
        except pd.errors.EmptyDataError as exc:
            raise ValueError(f"{file_path} is empty or malformed") from exc

        keep_columns = [TIMESTAMP_COLUMN] + self.core_channels
        missing = [col for col in keep_columns if col not in df.columns]
        if missing:
            raise ValueError(
                f"{file_path.name} is missing required sensor channels: {missing}"
            )
        trimmed = df[keep_columns].copy()
        trimmed[TIMESTAMP_COLUMN] = trimmed[TIMESTAMP_COLUMN].astype(str).str.strip()
        for column in self.core_channels:
            trimmed[column] = pd.to_numeric(trimmed[column], errors="coerce")
        trimmed = trimmed.dropna(subset=[TIMESTAMP_COLUMN]).reset_index(drop=True)
        return trimmed

    def load_all(self) -> Dict[str, pd.DataFrame]:
        """Load all datasets and return a dictionary keyed by filename."""
        datasets = {}
        for file_path in self.list_files():
            datasets[file_path.name] = self.load_file(file_path)
        return datasets

    def iter_datasets(self) -> Iterator[LoadedDataset]:
        """Yield one dataset at a time to keep memory usage low."""
        for file_path in self.list_files():
            df = self.load_file(file_path)
            yield LoadedDataset(
                path=file_path,
                label=self._infer_label(file_path),
                dataframe=df,
            )
