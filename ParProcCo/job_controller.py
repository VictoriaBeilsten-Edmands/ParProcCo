from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import List

from job_scheduler import JobScheduler


class SlicerInterface:

    def slice(self, input_data_file: Path, number_jobs: int, stop: int = None) -> List[slice]:
        """Takes an input data file and returns a list of slice parameters."""
        raise NotImplementedError


class AggregatorInterface:

    def aggregate(self, total_slices: int, aggregation_output_dir: Path, output_data_files: List[Path]) -> Path:
        """Aggregates data from multiple output files into one"""
        raise NotImplementedError


class JobController:

    def __init__(self, working_directory: str, cluster_output_dir: Path, project: str, queue: str, cpus: int = 16,
                 timeout: timedelta = timedelta(hours=2)):
        """JobController is used to coordinate cluster job submissions with JobScheduler"""

        self.cpus = cpus
        self.cluster_output_dir = Path(cluster_output_dir)
        self.data_aggregator: AggregatorInterface
        self.data_slicer: SlicerInterface
        self.project = project
        self.queue = queue
        self.scheduler: JobScheduler = None
        self.timeout = timeout
        self.working_directory = Path(working_directory)

    def run(self, data_slicer: SlicerInterface, data_aggregator: AggregatorInterface, input_path: Path,
            number_jobs: int, processing_script: Path) -> Path:
        self.data_slicer = data_slicer
        self.data_aggregator = data_aggregator
        slice_params = self.data_slicer.slice(input_path, number_jobs)

        self.scheduler = JobScheduler(self.working_directory, self.cluster_output_dir, self.project, self.queue,
                                      self.cpus, self.timeout)
        success = self.scheduler.run(processing_script, input_path, slice_params)
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
