from datetime import timedelta
from pathlib import Path
from job_scheduler import JobScheduler


class JobController:

    def __init__(self, working_directory, cluster_output_dir, project, priority, cpus=16, timeout=timedelta(hours=2)):
        """
        JobController is used to coordinate cluster job submissions with JobScheduler
        Args:
            working_directory (pathlib.Path or str): working directory
            cluster_output_dir (pathlib.Path or str): cluster output directory
            project (str): project name
            priority (str): name of queue to submit to
            cpus (int, optional): Number of CPUs to request. Defaults to 16.
            timeout (int, optional): Timeout for cluster jobs in minutes. Defaults to 180 (2 hours).
        """

        self.working_directory = Path(working_directory)
        self.cluster_output_dir = Path(cluster_output_dir)
        self.project = project
        self.priority = priority
        self.cpus = cpus
        self.timeout = timeout
        self.scheduler = None
        self.data_chunker = None

    def run(self, data_chunker, data_to_split, processing_script):
        self.data_chunker = data_chunker
        input_paths = self.data_chunker.chunk(self.working_directory, data_to_split)

        self.scheduler = JobScheduler(self.working_directory, self.cluster_output_dir, self.project, self.priority,
                                      self.cpus, self.timeout)
        self.scheduler.run(processing_script, input_paths, log_path=None)

        self.rerun_killed_jobs(processing_script)
        aggregated_data_path = self.aggregate_data()
        return aggregated_data_path

    def aggregate_data(self):
        if self.scheduler.get_success():
            chunked_results = self.scheduler.get_output_paths()
            aggregated_data_path = self.data_chunker.aggregate(self.cluster_output_dir, chunked_results)
            print(f"Processing complete. Aggregated results: {aggregated_data_path}\n")
            return aggregated_data_path
        else:
            raise RuntimeError(f"Not all jobs were successful. Aggregation not performed\n")

    def rerun_killed_jobs(self, processing_script):
        if not self.scheduler.get_success():
            job_history = self.scheduler.job_history
            failed_jobs = [job_info for job_info in job_history[0].values() if job_info["final_state"] != "SUCCESS"]

            if any(self.scheduler.job_completion_status.values()):
                killed_jobs = self.filter_killed_jobs(failed_jobs)
                killed_jobs_inputs = [job["input_path"] for job in killed_jobs]
                self.scheduler.resubmit_jobs(processing_script, killed_jobs_inputs)
            else:
                raise RuntimeError(f"All jobs failed\n")

    def filter_killed_jobs(self, jobs):
        killed_jobs = [job for job in jobs if job["info"].terminating_signal == "SIGKILL"]
        return killed_jobs
