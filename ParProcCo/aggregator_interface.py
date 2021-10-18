from __future__ import annotations

from pathlib import Path
from typing import List


class AggregatorInterface:

    def aggregate(self, total_slices: int, aggregation_output_dir: Path, output_data_files: List[Path]) -> Path:
        """Aggregates data from multiple output files into one"""
        raise NotImplementedError
