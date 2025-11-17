"""Reporting utilities for persisting intermediate analysis artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


class ReportWriter:
    """Handles writing structured reports to disk."""

    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_json(self, payload: Dict[str, Any], filename: str) -> Path:
        """Write a dictionary as JSON with indentation."""
        target = self.output_dir / filename
        with target.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        return target

    def write_markdown(self, text: str, filename: str) -> Path:
        """Write a Markdown text file."""
        target = self.output_dir / filename
        target.write_text(text, encoding="utf-8")
        return target
