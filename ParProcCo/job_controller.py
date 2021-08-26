from datetime import timedelta
from pathlib import Path
from typing import Any, List
from job_scheduler import JobScheduler


class JobController:

    def __init__(self, working_directory: str, cluster_output_dir: Path, project: str, queue: str, cpus: int = 16, timeout: timedelta = timedelta(hours=2)):
        """
        JobController is used to coordinate cluster job submissions with JobScheduler
        Args:
            working_directory (pathlib.Path or str): working directory
            cluster_output_dir (pathlib.Path or str): cluster output directory
            project (str): project name
            queue (str): name of queue to submit to
            cpus (int, optional): Number of CPUs to request. Defaults to 16.
            timeout (int, optional): Timeout for cluster jobs in minutes. Defaults to 180 (2 hours).
        """

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

        self.rerun_killed_jobs(processing_script)
        aggregated_file_path = self.aggregate_data()
        return aggregated_file_path

    def aggregate_data(self) -> Path:
        if self.scheduler.get_success():
            sliced_results = self.scheduler.get_output_paths()
            aggregated_file_path = self.data_aggregator.aggregate(self.cluster_output_dir, sliced_results)
            print(f"Processing complete. Aggregated results: {aggregated_file_path}\n")
            return aggregated_file_path
        else:
            raise RuntimeError(f"Not all jobs were successful. Aggregation not performed\n")

    def rerun_killed_jobs(self, processing_script: Path):
        if not self.scheduler.get_success():
            job_history = self.scheduler.job_history
            failed_jobs = [job_info for job_info in job_history[0].values() if job_info["final_state"] != "SUCCESS"]

            if any(self.scheduler.job_completion_status.values()):
                killed_jobs = self.filter_killed_jobs(failed_jobs)
                killed_jobs_inputs = [job["input_path"] for job in killed_jobs]
                if not all(x == killed_jobs_inputs[0] for x in killed_jobs_inputs):
                    raise RuntimeError(f"input paths in killed_jobs must all be the same\n")
                slice_params = [job["slice_param"] for job in killed_jobs]
                self.scheduler.resubmit_jobs(processing_script, killed_jobs_inputs[0], slice_params)
            else:
                raise RuntimeError(f"All jobs failed\n")

    def filter_killed_jobs(self, jobs: List) -> List:
        killed_jobs = [job for job in jobs if job["info"].terminating_signal == "SIGKILL"]
        return killed_jobs
