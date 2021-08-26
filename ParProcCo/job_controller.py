from datetime import timedelta
from pathlib import Path
from typing import Any
from job_scheduler import JobScheduler


class JobController:

    def __init__(self, working_directory: str, cluster_output_dir: Path, project: str, queue: str, cpus: int = 16, timeout: timedelta = timedelta(hours=2)):
        """JobController is used to coordinate cluster job submissions with JobScheduler"""

        self.working_directory = Path(working_directory)
        self.cluster_output_dir = Path(cluster_output_dir)
        self.project = project
        self.queue = queue
        self.cpus = cpus
        self.timeout = timeout
        self.scheduler: JobScheduler = None
        self.data_slicer: Any = None
        self.data_aggregator: Any = None

    def run(self, data_slicer: Any, data_aggregator: Any, input_path: Path, number_jobs: int, processing_script: Path) -> Path:
        self.data_slicer = data_slicer
        self.data_aggregator = data_aggregator
        slice_params = self.data_slicer.slice(input_path, number_jobs)

        self.scheduler = JobScheduler(self.working_directory, self.cluster_output_dir, self.project, self.queue,
                                      self.cpus, self.timeout)
        self.scheduler.run(processing_script, input_path, slice_params)

        self.scheduler.rerun_killed_jobs(processing_script)
        aggregated_file_path = self.aggregate_data()
        return aggregated_file_path

    def aggregate_data(self) -> Path:
        if self.scheduler.get_success():
            sliced_results = self.scheduler.get_output_paths()
            aggregated_file_path = self.data_aggregator.aggregate(self.cluster_output_dir, sliced_results)
            return aggregated_file_path
        else:
            raise RuntimeError(f"Not all jobs were successful. Aggregation not performed\n")
