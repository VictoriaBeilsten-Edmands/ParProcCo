import logging
import os
import stat
import sys
from pathlib import Path
from datetime import timedelta
from datetime import datetime

import drmaa2 as drmaa2
from drmaa2 import JobSession, JobTemplate, Drmaa2Exception


class TemporarilyLogToFile:

    def __init__(self, logger, path=None):
        self.logger = logger
        self.log_path = path
        if self.log_path is not None:
            self.file_handler = logging.FileHandler(self.log_path, "a+")
            self.logger.addHandler(self.file_handler)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        if self.log_path is not None:
            self.file_handler.close()
            self.logger.removeHandler(self.file_handler)
            with open(self.log_path, "a+") as fp:
                fp.write("\n")


class JobScheduler:

    def __init__(self, working_directory, cluster_output_dir, project, priority, cpus=16, timeout=timedelta(hours=2)):
        """
        JobScheduler can be used for cluster job submissions
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
        self.project = self.check_project_list(project)
        self.priority = self.check_queue_list(priority)
        self.cpus = cpus
        self.timeout = timeout
        self.logger = logging.getLogger()
        self.batch_number = 0
        self.output_paths = []
        self.start_time = datetime.now()
        self.job_history = {}
        self.job_completion_status = {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        pass

    def check_queue_list(self, priority):
        if priority is None:
            raise ValueError(f"ValueError. priority is {priority}")
        elif priority == "":
            raise ValueError(f"ValueError. priority is empty string")
        priority = priority.lower()
        q_proc = os.popen('qconf -sql')
        q_name_list = q_proc.read().split()
        q_proc.close()
        if priority in q_name_list:
            return priority
        else:
            raise ValueError(f"ValueError. priority {priority} not in queue list: {q_name_list}\n")

    def check_project_list(self, project):
        if project is None:
            raise ValueError(f"ValueError. project is {project}")
        elif project == "":
            raise ValueError(f"ValueError. project is empty string")
        prj_proc = os.popen('qconf -sprjl')
        prj_name_list = prj_proc.read().split()
        prj_proc.close()
        if project in prj_name_list:
            return project
        else:
            raise ValueError(f"ValueError. {project} not in list of project names: {prj_name_list}\n")

    def check_jobscript(self, jobscript):
        if not Path(jobscript).exists():
            raise FileNotFoundError(f"FileNotFoundError. {jobscript} does not exist\n")
        elif not bool(os.stat(jobscript).st_mode & stat.S_IXUSR):
            raise PermissionError(f"PermissionError. {jobscript} is not executable by user\n")
        else:
            return jobscript

    def get_output_paths(self):
        return self.output_paths

    def get_success(self):
        return all(self.job_completion_status.values())

    def get_failed_jobs(self):
        return {k: v for k, v in self.job_completion_status.items() if not v}

    def get_job_history(self):
        return self.job_history

    def timestamp_ok(self, output):
        mtime = datetime.fromtimestamp(output.stat().st_mtime)
        if mtime > self.start_time:
            return True
        return False

    def run(self, jobscript, input_paths, log_path=None):
        self.job_history[self.batch_number] = {}
        self.job_completion_status = {str(input_path): False for input_path in input_paths}
        self.run_and_monitor(jobscript, input_paths, log_path)

    def run_and_monitor(self, jobscript, input_paths, log_path=None):
        self.check_jobscript(jobscript)
        self.job_details = []
        self.log_path = log_path

        session = JobSession()  # Automatically destroyed when it is out of scope
        self.run_jobs(session, jobscript, input_paths)
        self.wait_for_jobs(session)
        self.report_job_info()

    def run_jobs(self, session, jobscript, input_paths):
        logging.debug(f"Running job on cluster for {input_paths}")
        try:
            # Run all input paths in parallel:
            for input_path in input_paths:
                template = self.create_template(input_path, jobscript)
                logging.debug(f"Submitting drmaa job with file {input_path}")
                job = session.run_job(template)
                self.job_details.append([job, input_path, Path(template.output_path)])
                logging.debug(f"drmaa job for file {input_path} has been submitted with id {job.id}")
        except Drmaa2Exception:
            logging.error(f"Drmaa exception", exc_info=True)
            raise
        except Exception:
            logging.error(f"Unknown error occurred running drmaa job", exc_info=True)
            raise

    def create_template(self, input_path, jobscript, job_name="job_scheduler_testing"):
        if not self.cluster_output_dir.exists():
            logging.debug(f"Making directory {self.cluster_output_dir}")
            self.cluster_output_dir.mkdir(exist_ok=True, parents=True)
        else:
            logging.debug(f"Directory {self.cluster_output_dir} already exists")

        args = [str(input_path)]
        output_file = f"out_{input_path.stem}.txt"
        err_file = f"err_{input_path.stem}.txt"
        output_fp = str(self.cluster_output_dir / output_file)
        err_fp = str(self.cluster_output_dir / err_file)
        self.output_paths.append(output_fp)

        jt = JobTemplate({
            "job_name": job_name,
            "job_category": self.project,
            "remote_command": str(self.working_directory / jobscript),
            "min_slots": self.cpus,
            "args": args,
            "working_directory": str(self.working_directory),
            "output_path": output_fp,
            "error_path": err_fp,
            "queue_name": self.priority,
            "implementation_specific": {
                "uge_jt_pe": f"smp",
            },
        })
        return jt

    def wait_for_jobs(self, session):
        with TemporarilyLogToFile(self.logger, self.log_path):
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

    def report_job_info(self):
        # Iterate through jobs with logging to check individual job outcomes
        for job, filename, output in self.job_details:
            with TemporarilyLogToFile(self.logger, self.log_path):
                logging.debug(f"Retrieving info for drmaa job {job.id} for file {filename}")
                try:
                    js = job.get_state()[0]  # Returns job state and job substate (always seems to be None)
                    ji = job.get_info()
                    exit_stat = ji.exit_status
                    j_state = ji.job_state
                    term_sig = ji.terminating_signal

                except Exception:
                    logging.error(f"Failed to get job information for job {job.id} processing file", exc_info=True)
                    raise

                # Check job states against expected possible options:
                if js == drmaa2.JobState.UNDETERMINED:  # Lost contact?
                    self.job_history[self.batch_number][job.id] = {"info": ji, "state": js, "exit_stat": exit_stat,
                                                                   "j_state": j_state, "term_sig": term_sig,
                                                                   "input_path": filename,
                                                                   "final_state": "UNDETERMINED"}
                    logging.warning(f"Job state undetermined for processing file {filename}. job info: {ji}")

                elif js == drmaa2.JobState.FAILED:
                    self.job_history[self.batch_number][job.id] = {"info": ji, "state": js, "exit_stat": exit_stat,
                                                                   "j_state": j_state, "term_sig": term_sig,
                                                                   "input_path": filename, "final_state": "FAILED"}
                    logging.error(
                        f"drmaa job {job.id} processing file filename failed."
                        f" Terminating signal: {ji.terminating_signal}."
                    )

                elif not output.exists():
                    self.job_history[self.batch_number][job.id] = {"info": ji, "state": js, "exit_stat": exit_stat,
                                                                   "j_state": j_state, "term_sig": term_sig,
                                                                   "input_path": filename, "final_state": "NO_OUTPUT"}
                    logging.error(
                        f"drmaa job {job.id} processing file {filename} has not created output file {output}"
                        f" Terminating signal: {ji.terminating_signal}."
                    )

                elif not self.timestamp_ok(output):
                    self.job_history[self.batch_number][job.id] = {"info": ji, "state": js, "exit_stat": exit_stat,
                                                                   "j_state": j_state, "term_sig": term_sig,
                                                                   "input_path": filename,
                                                                   "final_state": "OLD_OUTPUT_FILE"}
                    logging.error(
                        f"drmaa job {job.id} processing file {filename} has not created a new output file {output}. "
                        f"Terminating signal: {ji.terminating_signal}."
                    )

                elif js == drmaa2.JobState.DONE:
                    self.job_completion_status[str(filename)] = True
                    self.job_history[self.batch_number][job.id] = {"info": ji, "state": js, "exit_stat": exit_stat,
                                                                   "j_state": j_state, "term_sig": term_sig,
                                                                   "input_path": filename, "final_state": "SUCCESS"}
                    logging.info(
                        f"Job {job.id} processing file {filename} completed successfully after {ji.wallclock_time}. "
                        f"CPU time={timedelta(seconds=float(ji.cpu_time))}, slots={ji.slots}"
                    )
                else:
                    self.job_history[self.batch_number][job.id] = {"info": ji, "state": js, "exit_stat": exit_stat,
                                                                   "j_state": j_state, "term_sig": term_sig,
                                                                   "input_path": filename, "final_state": "UNSPECIFIED"}
                    logging.error(f"Unexpected job state for file {filename}, job info: {ji}")

    def resubmit_jobs(self, jobscript, jobs):
        # failed_jobs list is list of lists [JobInfo, input_path, output_path]
        self.batch_number += 1
        self.job_history[self.batch_number] = {}
        self.run(jobscript, jobs)
