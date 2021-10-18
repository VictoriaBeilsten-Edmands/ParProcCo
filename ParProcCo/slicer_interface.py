from __future__ import annotations

from typing import List


class SlicerInterface:

    def slice(self, number_jobs: int, stop: int = None) -> List[slice]:
        """Takes an input data file and returns a list of slice parameters."""
        raise NotImplementedError
