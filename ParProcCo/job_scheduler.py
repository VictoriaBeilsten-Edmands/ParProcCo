from __future__ import annotations

import logging
import os
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Union

import drmaa2 as drmaa2
from drmaa2 import Drmaa2Exception, JobSession, JobTemplate


@dataclass
class StatusInfo:
    '''Class for keeping track of job status.'''
    info: drmaa2.JobInfo
    state: drmaa2.JobState
    input_path: Path
    slice_param: List[str]
    terminating_signal: Union[str, None] = None
    final_state: Union[str, None] = None


class JobScheduler:

    def __init__(self, working_directory: str, cluster_output_dir: Path, project: str, queue: str, cpus: int = 16,
                 timeout: timedelta = timedelta(hours=2)):
        """JobScheduler can be used for cluster job submissions"""
        self.working_directory = Path(working_directory)
        self.cluster_output_dir = Path(cluster_output_dir)
        self.project = self.check_project_list(project)
        self.queue = self.check_queue_list(queue)
        self.cpus = cpus
        self.timeout = timeout
        self.batch_number = 0
        self.output_paths: List[Path] = []
        self.start_time = datetime.now()
        self.job_history: Dict[int, Dict[int, StatusInfo]] = {}
        self.job_completion_status: Dict[str, bool] = {}

    def check_queue_list(self, queue: str) -> str:
        if not queue:
            raise ValueError(f"queue must be non-empty string")
        queue = queue.lower()
        with os.popen('qconf -sql') as q_proc:
            q_name_list = q_proc.read().split()
        if queue in q_name_list:
            return queue
        else:
            raise ValueError(f"queue {queue} not in queue list: {q_name_list}\n")

    def check_project_list(self, project: str) -> str:
        if not project:
            raise ValueError(f"project must be non-empty string")
        with os.popen('qconf -sprjl') as prj_proc:
            prj_name_list = prj_proc.read().split()
        if project in prj_name_list:
            return project
        else:
            raise ValueError(f"{project} must be in list of project names: {prj_name_list}\n")

    def check_jobscript(self, jobscript: Path) -> Path:
        if not jobscript.is_file():
            raise FileNotFoundError(f"{jobscript} does not exist\n")

        if not (os.access(jobscript, os.R_OK) and os.access(jobscript, os.X_OK)):
            raise PermissionError(f"{jobscript} must be readable and executable by user\n")

        try:
            js = jobscript.open()
            js.close()
        except IOError:
            logging.error(f"{jobscript} cannot be opened\n")
            raise

        else:
            return jobscript

    def get_output_paths(self) -> List[Path]:
        return self.output_paths

    def get_success(self) -> bool:
        return all(self.job_completion_status.values())

    def timestamp_ok(self, output: Path) -> bool:
        mod_time = datetime.fromtimestamp(output.stat().st_mtime)
        if mod_time > self.start_time:
            return True
        return False

    def run(self, jobscript: Path, input_path: Path, slices: List[slice]) -> bool:
        self.job_history[self.batch_number] = {}
        self.job_completion_status = {f"{s.start}:{s.stop}:{s.step}": False for s in slices}
        self._run_and_monitor(jobscript, input_path, slices)
        return self.get_success()

    def _run_and_monitor(self, jobscript: Path, input_path: Path, slices: List[slice]) -> None:
        jobscript = self.check_jobscript(jobscript)
        self.job_details: List[List] = []

        session = JobSession()  # Automatically destroyed when it is out of scope
        self._run_jobs(session, jobscript, input_path, slices)
        self._wait_for_jobs(session)
        self._report_job_info()

    def _run_jobs(self, session: JobSession, jobscript: Path, input_path: Path, slices: List[slice]) -> None:
        logging.debug(f"Running jobs on cluster for {input_path}")
        slice_params = [[f"{s.start}:{s.stop}:{s.step}"] for s in slices]
        try:
            # Run all input paths in parallel:
            for i, slice_param in enumerate(slice_params):
                template = self._create_template(input_path, jobscript, slice_param, i)
                logging.debug(f"Submitting drmaa job with file {input_path}")
                job = session.run_job(template)
                self.job_details.append([job, input_path, slice_param, Path(template.output_path)])
                logging.debug(f"drmaa job for file {input_path} has been submitted with id {job.id}")
        except Drmaa2Exception:
            logging.error(f"Drmaa exception", exc_info=True)
            raise
        except Exception:
            logging.error(f"Unknown error occurred running drmaa job", exc_info=True)
            raise

    def _create_template(self, input_path: Path, jobscript: Path, slice_param: List[str], i: int,
                         job_name: str = "job_scheduler_testing") -> JobTemplate:
        if not self.cluster_output_dir.exists():
            logging.debug(f"Making directory {self.cluster_output_dir}")
            self.cluster_output_dir.mkdir(exist_ok=True, parents=True)
        else:
            logging.debug(f"Directory {self.cluster_output_dir} already exists")

        output_file = f"out_{input_path.stem}_{i}.txt"
        err_file = f"err_{input_path.stem}_{i}.txt"
        output_fp = str(self.cluster_output_dir / output_file)
        err_fp = str(self.cluster_output_dir / err_file)
        self.output_paths.append(Path(output_fp))
        args = [f"--input_path", str(input_path), f"--output_path", str(output_fp), f"-I"] + slice_param

        jt = JobTemplate({
            "job_name": job_name,
            "job_category": self.project,
            "remote_command": str(self.working_directory / jobscript),
            "min_slots": self.cpus,
            "args": args,
            "working_directory": str(self.working_directory),
            "output_path": output_fp,
            "error_path": err_fp,
            "queue_name": self.queue,
            "implementation_specific": {
                "uge_jt_pe": f"smp",
            },
        })
        return jt

    def _wait_for_jobs(self, session: JobSession) -> None:
        try:
            job_list = [job_info[0] for job_info in self.job_details]
            # Wait for jobs to start (timeout shouldn't include queue time)
            job_list_str = ", ".join([str(job.id) for job in job_list])
            logging.info(f"Waiting for jobs to start: {job_list_str}")
            session.wait_all_started(job_list)
            logging.info(f"Jobs started, waiting for jobs: {job_list_str}")
            session.wait_all_terminated(job_list, int(round(self.timeout.total_seconds())))
            jobs_running = False
            for job in job_list:
                if job.get_state()[0] == drmaa2.JobState.RUNNING:
                    logging.info(f"Job {job.id} timed out. Terminating job now.")
                    jobs_running = True
                    job.terminate()
                    print(f"terminating job {job.id}")
            if jobs_running:
                # Termination takes some time, wait a max of 2 mins
                session.wait_all_terminated(job_list, 120)
        except Drmaa2Exception:
            logging.error(f"Drmaa exception", exc_info=True)
        except Exception:
            logging.error(f"Unknown error occurred running drmaa job", exc_info=True)

    def _report_job_info(self) -> None:
        # Iterate through jobs with logging to check individual job outcomes
        for job, filename, slice_param, output in self.job_details:
            logging.debug(f"Retrieving info for drmaa job {job.id} for file {filename}")
            try:
                js = job.get_state()[0]  # Returns job state and job substate (always seems to be None)
                ji = job.get_info()

            except Exception:
                logging.error(f"Failed to get job information for job {job.id} processing file", exc_info=True)
                raise

            # Check job states against expected possible options:
            status_info = StatusInfo(ji, js, filename, slice_param)
            if js == drmaa2.JobState.UNDETERMINED:  # Lost contact?
                status_info.final_state = "UNDETERMINED"
                logging.warning(f"Job state undetermined for processing file {filename}. job info: {ji}")

            elif js == drmaa2.JobState.FAILED:
                status_info.final_state = "FAILED"
                logging.error(
                    f"drmaa job {job.id} processing file filename failed."
                    f" Terminating signal: {ji.terminating_signal}."
                )

            elif not output.exists():
                status_info.final_state = "NO_OUTPUT"
                logging.error(
                    f"drmaa job {job.id} processing file {filename} with slice parameters {slice_param} has not created"
                    f" output file {output}"
                    f" Terminating signal: {ji.terminating_signal}."
                )

            elif not self.timestamp_ok(output):
                status_info.final_state = "OLD_OUTPUT_FILE"
                logging.error(
                    f"drmaa job {job.id} processing file {filename} with slice parameters {slice_param} has not created"
                    f" a new output file {output}"
                    f"Terminating signal: {ji.terminating_signal}."
                )

            elif js == drmaa2.JobState.DONE:
                self.job_completion_status["".join(slice_param)] = True
                status_info.final_state = "SUCCESS"
                logging.info(
                    f"Job {job.id} processing file {filename} with slice parameters {slice_param} completed"
                    f" successfully after {ji.wallclock_time}."
                    f" CPU time={timedelta(seconds=float(ji.cpu_time))}, slots={ji.slots}"
                )
            else:
                status_info.final_state = "UNSPECIFIED"
                logging.error(
                    f"Unexpected job state for file {filename} with slice parameters {slice_param}, job info: {ji}"
                )

            self.job_history[self.batch_number][job.id] = status_info

    def resubmit_jobs(self, jobscript: Path, input_path: Path, slice_params: List[List[str]]) -> None:
        # failed_jobs list is list of lists [JobInfo, input_path, output_path]
        self.batch_number += 1
        slices = [slice(*[int(x) for x in ("".join(slice_param)).split(":")]) for slice_param in slice_params]
        self.run(jobscript, input_path, slices)

    def filter_killed_jobs(self, jobs: List[drmaa2.Job]) -> List[drmaa2.Job]:
        killed_jobs = [job for job in jobs if job["info"].terminating_signal == "SIGKILL"]
        return killed_jobs

    def rerun_killed_jobs(self, jobscript: Path):
        job_history = self.job_history
        if all(self.job_completion_status.values()):
            warnings.warn("No failed jobs")
        elif any(self.job_completion_status.values()):
            failed_jobs = [job_info for job_info in job_history[0].values() if job_info.final_state != "SUCCESS"]
            killed_jobs = self.filter_killed_jobs(failed_jobs)
            killed_jobs_inputs = [job["input_path"] for job in killed_jobs]
            if not all(x == killed_jobs_inputs[0] for x in killed_jobs_inputs):
                raise RuntimeError(f"input paths in killed_jobs must all be the same\n")
            slice_params = [job["slice_param"] for job in killed_jobs]
            self.resubmit_jobs(jobscript, killed_jobs_inputs[0], slice_params)
        else:
            raise RuntimeError(f"All jobs failed\n")
