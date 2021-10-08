from __future__ import annotations

import os
from datetime import timedelta
from pathlib import Path
from typing import List, Union

from job_scheduler import JobScheduler


def check_location(location: Union[Path, str]) -> Path:
    location_path = Path(location)
    if Path("/dls") in location_path.parents or Path("/home") in location_path.parents or Path("/dls_sw") in location_path.parents:
        return location_path
    raise ValueError(f"{location_path} must be located within /dls, 'dls_sw or /home")


def get_absolute_path(filename: Union[Path, str]) -> str:
    python_path = os.environ['PYTHONPATH'].split(os.pathsep)
    for search_path in python_path:
        for root, dir, files in os.walk(search_path):
            if filename in files:
                return os.path.join(root, filename)

    return os.path.abspath(filename)


class SlicerInterface:

    def slice(self, number_jobs: int, stop: int = None) -> List[slice]:
        """Takes an input data file and returns a list of slice parameters."""
        raise NotImplementedError


class AggregatorInterface:

    def aggregate(self, total_slices: int, aggregation_output_dir: Path, output_data_files: List[Path]) -> Path:
        """Aggregates data from multiple output files into one"""
        raise NotImplementedError


class JobController:

    def __init__(self, cluster_output_dir_name: str, project: str, queue: str, timeout: timedelta = timedelta(hours=2)):
        """JobController is used to coordinate cluster job submissions with JobScheduler"""

        self.working_directory: Path = check_location(os.getcwd())

        self.cluster_output_dir: Path = self.working_directory / cluster_output_dir_name
        self.data_aggregator: AggregatorInterface
        self.data_slicer: SlicerInterface
        self.project = project
        self.queue = queue
        self.scheduler: JobScheduler = None
        self.timeout = timeout

    def run(self, data_slicer: SlicerInterface, data_aggregator: AggregatorInterface, number_jobs: int,
            processing_script: Path, memory: str = "4G", cores: int = 6, jobscript_args: List = None,
            job_name: str = "ParProcCo") -> Path:
        self.data_slicer = data_slicer
        self.data_aggregator = data_aggregator
        processing_script = check_location(get_absolute_path(processing_script))
        slice_params = self.data_slicer.slice(number_jobs)
        if jobscript_args is None:
            jobscript_args = []

        self.scheduler = JobScheduler(self.working_directory, self.cluster_output_dir, self.project, self.queue,
                                      self.timeout)
        success = self.scheduler.run(processing_script, slice_params, memory, cores, jobscript_args, job_name)
        if not success:
            self.scheduler.rerun_killed_jobs(processing_script)
        aggregated_file_path = self.aggregate_data(number_jobs)
        return aggregated_file_path

    def aggregate_data(self, number_jobs: int) -> Path:
        if self.scheduler.get_success():
            sliced_results = self.scheduler.get_output_paths()
            aggregated_file_path = self.data_aggregator.aggregate(number_jobs, self.cluster_output_dir, sliced_results)
            return aggregated_file_path
        else:
            raise RuntimeError(f"Not all jobs were successful. Aggregation not performed\n")
