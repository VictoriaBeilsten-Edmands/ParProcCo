from __future__ import annotations

from pathlib import Path
from typing import List

from job_controller import SlicerInterface


class SimpleDataSlicer(SlicerInterface):

    def __init__(self):
        pass

    def slice(self, input_data_file: Path, number_jobs: int, stop: int = None) -> List:
        """Overrides SlicerInterface.slice"""
        if type(number_jobs) is not int:
            raise TypeError(f"number_jobs is {type(number_jobs)}, should be int\n")

        file_length = sum(1 for line in open(input_data_file))

        if stop is None:
            stop = file_length
        elif type(stop) is not int:
            raise TypeError(f"stop is {type(stop)}, should be int\n")
        else:
            stop = min(stop, file_length)

        number_jobs = min(stop, number_jobs)

        slice_params = [[f"{i}:{stop}:{number_jobs}"]
                        for i in range(number_jobs)]
        return slice_params
