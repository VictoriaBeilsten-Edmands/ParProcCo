import drmaa2 as drmaa2
from datetime import datetime
from datetime import timedelta
import getpass
import logging
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from typing import List, Tuple

from job_scheduler import JobScheduler


def setup_data_files(working_directory: str, cluster_output_dir: Path) -> Tuple[Path, List[Path], List[str], List[List[str]]]:
    file_name = "test_raw_data.txt"
    input_file_path = Path(working_directory) / file_name
    with open(input_file_path, "w") as f:
        f.write("0\n1\n2\n3\n4\n5\n6\n7\n8\n9\n10\n")
        slice_params = []
    for i in range(4):
        slice_params.append([f"{i}:8:4"])

    output_file_paths = [Path(cluster_output_dir) / f"out_{input_file_path.stem}_{i}.txt" for i in range(4)]
    output_nums = ["0\n8\n", "2\n10\n", "4\n12\n", "6\n14\n"]
    return input_file_path, output_file_paths, output_nums, slice_params


def setup_jobscript(working_directory: str) -> Path:
    jobscript = Path(working_directory) / "test_script.py"
    with open(jobscript, "x") as f:
        jobscript_lines = """
#!/usr/bin/env python3

import argparse


def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", help="str: path to input file", type=str)
    parser.add_argument("--output_path", help="str: path to output file", type=str)
    parser.add_argument("-I", help="str: slice selection of images per input file (as start:stop:step)")
    return parser


def check_args(args):
    empty_fields = [k for k, v in vars(args).items() if v is None]
    if len(empty_fields) > 0:
        raise ValueError(f"Missing arguments: {empty_fields}")


def write_lines(input_path, output_path, images):
    start, stop, step = images.split(":")
    start = int(start)
    stop = int(stop)
    step = int(step)
    with open(input_path, "r") as in_f:
        for i, line in enumerate(in_f):
            if i >= stop:
                break

            elif i >= start and ((i - start) % step == 0):
                doubled = int(line.strip("\\n")) * 2
                doubled_str = f"{doubled}\\n"
                with open(output_path, "a+") as out_f:
                    out_f.write(doubled_str)


if __name__ == '__main__':
    '''
    $ python jobscript.py --input_path input_path --output_path output_path -I slice_param
    '''
    parser = setup_parser()
    args = parser.parse_args()
    check_args(args)

    write_lines(args.input_path, args.output_path, args.I)
"""
        jobscript_lines = jobscript_lines.lstrip()
        f.write(jobscript_lines)
    os.chmod(jobscript, 0o777)
    return jobscript


class TestJobScheduler(unittest.TestCase):

    def setUp(self) -> None:
        current_user = getpass.getuser()
        tmp_dir = f"/dls/tmp/{current_user}/"
        self.base_dir = f"/dls/tmp/{current_user}/tests/"
        self.assertTrue(Path(tmp_dir).is_dir(), f"{tmp_dir} is not a directory")
        if not Path(self.base_dir).is_dir():
            logging.debug(f"Making directory {self.base_dir}")
            Path(self.base_dir).mkdir(exist_ok=True)

    def test_create_template_with_cluster_output_dir(self) -> None:
        with TemporaryDirectory(prefix='test_dir_', dir=self.base_dir) as working_directory:
            input_paths = Path('path/to/file.extension')
            cluster_output_dir = os.path.join(working_directory, 'cluster_output_dir')
            js = JobScheduler(working_directory, cluster_output_dir, project="b24", priority="medium.q")
            js.create_template(input_paths, "some_script.py", ["slice_param"], 1)
            cluster_output_dir_exists = os.path.exists(cluster_output_dir)
        self.assertTrue(cluster_output_dir_exists, msg="Cluster output directory was not created\n")

    def test_job_scheduler_runs(self) -> None:
        # create directory for test files
        with TemporaryDirectory(prefix='test_dir_', dir=self.base_dir) as working_directory:
            cluster_output_dir = Path(working_directory) / "cluster_output"

            input_path, output_paths, out_nums, slice_params = setup_data_files(working_directory, cluster_output_dir)
            jobscript = setup_jobscript(working_directory)

            # run jobs
            js = JobScheduler(working_directory, cluster_output_dir, "b24", "medium.q")
            js.run(jobscript, input_path, slice_params)

            # check output files
            for output_file, expected_nums in zip(output_paths, out_nums):

                with open(output_file, "r") as f:
                    file_content = f.read()

                self.assertTrue(os.path.exists(output_file), msg=f"Output file {output_file} was not created\n")
                self.assertEqual(expected_nums, file_content, msg=f"Output file {output_file} content was incorrect\n")

    def test_old_output_timestamps(self) -> None:
        # create directory for test files
        with TemporaryDirectory(prefix='test_dir_', dir=self.base_dir) as working_directory:
            cluster_output_dir = Path(working_directory) / "cluster_output"
            jobscript = setup_jobscript(working_directory)

            input_file_path, _, _, slice_params = setup_data_files(working_directory, cluster_output_dir)

            # run jobs
            js = JobScheduler(working_directory, cluster_output_dir, "b24", "medium.q")

            js.job_history[js.batch_number] = {}
            js.job_completion_status = {"".join(slice_param): False for slice_param in slice_params}
            js.check_jobscript(jobscript)
            js.job_details = []

            session = drmaa2.JobSession()  # Automatically destroyed when it is out of scope
            js.run_jobs(session, jobscript, input_file_path, slice_params)
            js.wait_for_jobs(session)
            js.start_time = datetime.now()
            js.report_job_info()

            job_stats = js.job_completion_status
            # check failure list
            self.assertFalse(js.get_success(), msg=f"JobScheduler.success is not False\n")
            self.assertFalse(any(job_stats.values()), msg=f"All jobs not failed:"
                             f"{js.job_completion_status.values()}\n")
            self.assertEqual(len(job_stats), 4,
                             msg=f"len(js.job_completion_status) is not 4. js.job_completion_status: {job_stats}\n")

    def test_get_all_queues(self) -> None:
        with os.popen('qconf -sql') as q_proc:
            q_name_list = q_proc.read().split()
        ms = drmaa2.MonitoringSession('ms-01')
        qi_list = ms.get_all_queues(q_name_list)
        self.assertEqual(len(qi_list), len(q_name_list))
        for qi in qi_list:
            q_name = qi.name
            self.assertTrue(q_name in q_name_list)

    def test_project_is_none(self) -> None:
        with TemporaryDirectory(prefix='test_dir_', dir=self.base_dir) as working_directory:
            cluster_output_dir = Path(working_directory) / "cluster_output"

            with self.assertRaises(ValueError) as context:
                JobScheduler(working_directory, cluster_output_dir, None, "medium.q")
            self.assertTrue("project must be non-empty string" in str(context.exception))

    def test_project_is_empty(self) -> None:
        with TemporaryDirectory(prefix='test_dir_', dir=self.base_dir) as working_directory:
            cluster_output_dir = Path(working_directory) / "cluster_output"

            with self.assertRaises(ValueError) as context:
                JobScheduler(working_directory, cluster_output_dir, "", "medium.q")
            self.assertTrue("project must be non-empty string" in str(context.exception))

    def test_check_project_list(self) -> None:
        with TemporaryDirectory(prefix='test_dir_', dir=self.base_dir) as working_directory:
            cluster_output_dir = Path(working_directory) / "cluster_output"

            js = JobScheduler(working_directory, cluster_output_dir, "b24", "medium.q")
            self.assertEqual(js.project, "b24")

    def test_bad_project_name(self) -> None:
        with TemporaryDirectory(prefix='test_dir_', dir=self.base_dir) as working_directory:
            cluster_output_dir = Path(working_directory) / "cluster_output"

            with self.assertRaises(ValueError) as context:
                JobScheduler(working_directory, cluster_output_dir, "bad_project_name", "medium.q")
            self.assertTrue("bad_project_name must be in list of project names" in str(context.exception))

    def test_check_queue_list(self):
        with TemporaryDirectory(prefix='test_dir_', dir=self.base_dir) as working_directory:
            cluster_output_dir = Path(working_directory) / "cluster_output"

            js = JobScheduler(working_directory, cluster_output_dir, "b24", "medium.q")
            self.assertEqual(js.priority, "medium.q")

    def test_uppercase_queue_list(self):
        with TemporaryDirectory(prefix='test_dir_', dir=self.base_dir) as working_directory:
            cluster_output_dir = Path(working_directory) / "cluster_output"

            js = JobScheduler(working_directory, cluster_output_dir, "b24", "MEDIUM.Q")
            self.assertEqual(js.priority, "medium.q")

    def test_queue_is_none(self) -> None:
        with TemporaryDirectory(prefix='test_dir_', dir=self.base_dir) as working_directory:
            cluster_output_dir = Path(working_directory) / "cluster_output"

            with self.assertRaises(ValueError) as context:
                JobScheduler(working_directory, cluster_output_dir, "b24", None)
            self.assertTrue("priority must be non-empty string" in str(context.exception))

    def test_queue_is_empty(self) -> None:
        with TemporaryDirectory(prefix='test_dir_', dir=self.base_dir) as working_directory:
            cluster_output_dir = Path(working_directory) / "cluster_output"

            with self.assertRaises(ValueError) as context:
                JobScheduler(working_directory, cluster_output_dir, "b24", "")
            self.assertTrue("priority must be non-empty string" in str(context.exception))

    def test_bad_queue_name(self) -> None:
        with TemporaryDirectory(prefix='test_dir_', dir=self.base_dir) as working_directory:
            cluster_output_dir = Path(working_directory) / "cluster_output"

            with self.assertRaises(ValueError) as context:
                JobScheduler(working_directory, cluster_output_dir, "b24", "bad_queue_name.q")
            self.assertTrue("priority bad_queue_name.q not in queue list" in str(context.exception))

    def test_job_times_out(self) -> None:
        with TemporaryDirectory(prefix='test_dir_', dir=self.base_dir) as working_directory:
            cluster_output_dir = Path(working_directory) / "cluster_output"

            jobscript = setup_jobscript(working_directory)
            with open(jobscript, "a+") as f:
                f.write("import time\ntime.sleep(5)\n")

            input_file_path, _, _, slice_params = setup_data_files(working_directory, cluster_output_dir)

            # run jobs
            js = JobScheduler(working_directory, cluster_output_dir, "b24", "medium.q",
                              timeout=timedelta(seconds=1))
            js.run(jobscript, input_file_path, slice_params)
            jh = js.job_history
            self.assertEqual(len(jh), 1, f"There should be one batch of jobs; job_history: {jh}\n")
            returned_jobs = jh[0]
            self.assertEqual(len(returned_jobs), 4)
            for job_id in returned_jobs:
                self.assertEqual(returned_jobs[job_id]["info"].exit_status, 137)
                self.assertEqual(returned_jobs[job_id]["info"].terminating_signal, "SIGKILL")
                self.assertEqual(returned_jobs[job_id]["info"].job_state, "FAILED")

    def test_bad_jobscript_name(self) -> None:
        with TemporaryDirectory(prefix='test_dir_', dir=self.base_dir) as working_directory:
            cluster_output_dir = Path(working_directory) / "cluster_output"

            js = JobScheduler(working_directory, cluster_output_dir, "b24", "medium.q")
            input_file_path, _, _, slice_params = setup_data_files(working_directory, cluster_output_dir)
            jobscript = Path(working_directory) / "bad_jobscript_name.sh"

            with self.assertRaises(FileNotFoundError) as context:
                js.run(jobscript, input_file_path, slice_params)

            self.assertTrue(f"{jobscript} does not exist\n" in str(context.exception))

    def test_insufficient_jobscript_permissions(self) -> None:
        with TemporaryDirectory(prefix='test_dir_', dir=self.base_dir) as working_directory:
            cluster_output_dir = Path(working_directory) / "cluster_output"

            js = JobScheduler(working_directory, cluster_output_dir, "b24", "medium.q")

            input_file_path, _, _, slice_params = setup_data_files(working_directory, cluster_output_dir)
            jobscript = Path(working_directory) / "test_bad_permissions.sh"
            f = open(jobscript, "x")
            f.close()
            os.chmod(jobscript, 0o666)

            with self.assertRaises(PermissionError) as context:
                js.run(jobscript, input_file_path, slice_params)

            self.assertTrue(f"{jobscript} must be readable and executable by user\n" in str(context.exception))

    def test_jobscript_cannot_be_opened(self) -> None:
        with TemporaryDirectory(prefix='test_dir_', dir=self.base_dir) as working_directory:
            cluster_output_dir = Path(working_directory) / "cluster_output"

            js = JobScheduler(working_directory, cluster_output_dir, "b24", "medium.q")
            input_file_path, _, _, slice_params = setup_data_files(working_directory, cluster_output_dir)
            jobscript = Path(working_directory) / "test_bad_read_permissions.sh"
            f = open(jobscript, "x")
            f.close()
            os.chmod(jobscript, 0o333)

            with self.assertRaises(PermissionError) as context:
                js.run(jobscript, input_file_path, slice_params)

            self.assertTrue(f"{jobscript} must be readable and executable by user\n" in str(context.exception))

    def test_get_output_paths(self) -> None:
        with TemporaryDirectory(prefix='test_dir_', dir=self.base_dir) as working_directory:
            cluster_output_dir = Path(working_directory) / "cluster_output"

            js = JobScheduler(working_directory, cluster_output_dir, "b24", "medium.q")
            js.output_paths = [cluster_output_dir / "out1.nxs", cluster_output_dir / "out2.nxs"]
            self.assertEqual(js.get_output_paths(), [cluster_output_dir / "out1.nxs", cluster_output_dir / "out2.nxs"])

    def test_get_success_all_true(self) -> None:
        with TemporaryDirectory(prefix='test_dir_', dir=self.base_dir) as working_directory:
            cluster_output_dir = Path(working_directory) / "cluster_output"

            js = JobScheduler(working_directory, cluster_output_dir, "b24", "medium.q")
            js.job_completion_status = {"0:8:4": True, "1:8:4": True}
            self.assertTrue(js.get_success())

    def test_get_success_all_false(self) -> None:
        with TemporaryDirectory(prefix='test_dir_', dir=self.base_dir) as working_directory:
            cluster_output_dir = Path(working_directory) / "cluster_output"

            js = JobScheduler(working_directory, cluster_output_dir, "b24", "medium.q")
            js.job_completion_status = {"0:8:4": False, "1:8:4": False}
            self.assertFalse(js.get_success())

    def test_get_success_true_false(self) -> None:
        with TemporaryDirectory(prefix='test_dir_', dir=self.base_dir) as working_directory:
            cluster_output_dir = Path(working_directory) / "cluster_output"

            js = JobScheduler(working_directory, cluster_output_dir, "b24", "medium.q")
            js.job_completion_status = {"0:8:4": False, "1:8:4": True}
            self.assertFalse(js.get_success())

    def test_get_failed_jobs_all(self) -> None:
        with TemporaryDirectory(prefix='test_dir_', dir=self.base_dir) as working_directory:
            cluster_output_dir = Path(working_directory) / "cluster_output"

            js = JobScheduler(working_directory, cluster_output_dir, "b24", "medium.q")
            js.job_completion_status = {"0:8:4": False, "1:8:4": False}
            self.assertEqual(js.get_failed_jobs(), {"0:8:4": False, "1:8:4": False})

    def test_get_failed_jobs_none(self) -> None:
        with TemporaryDirectory(prefix='test_dir_', dir=self.base_dir) as working_directory:
            cluster_output_dir = Path(working_directory) / "cluster_output"

            js = JobScheduler(working_directory, cluster_output_dir, "b24", "medium.q")
            js.job_completion_status = {"0:8:4": True, "1:8:4": True}
            self.assertEqual(js.get_failed_jobs(), {})

    def test_get_failed_jobs_some(self) -> None:
        with TemporaryDirectory(prefix='test_dir_', dir=self.base_dir) as working_directory:
            cluster_output_dir = Path(working_directory) / "cluster_output"

            js = JobScheduler(working_directory, cluster_output_dir, "b24", "medium.q")
            js.job_completion_status = {"0:8:4": True, "1:8:4": False}
            self.assertEqual(js.get_failed_jobs(), {"1:8:4": False})

    def test_get_job_history(self) -> None:
        with TemporaryDirectory(prefix='test_dir_', dir=self.base_dir) as working_directory:
            cluster_output_dir = Path(working_directory) / "cluster_output"

            js = JobScheduler(working_directory, cluster_output_dir, "b24", "medium.q")
            js.job_history = {0: {12: {"info": "0"}}, 1: {34: {"info": "1"}, 56: {"info": "2"}}}

            self.assertEqual(js.get_job_history(), {0: {12: {"info": "0"}}, 1: {34: {"info": "1"}, 56: {"info": "2"}}})

    def test_timestamp_ok_true(self) -> None:
        with TemporaryDirectory(prefix='test_dir_', dir=self.base_dir) as working_directory:
            cluster_output_dir = Path(working_directory) / "cluster_output"
            cluster_output_dir.mkdir(parents=True, exist_ok=True)
            filepath = cluster_output_dir / "out_0.nxs"

            js = JobScheduler(working_directory, cluster_output_dir, "b24", "medium.q")
            f = open(filepath, "x")
            f.close()

            self.assertTrue(js.timestamp_ok(filepath))

    def test_timestamp_ok_false(self) -> None:
        with TemporaryDirectory(prefix='test_dir_', dir=self.base_dir) as working_directory:
            cluster_output_dir = Path(working_directory) / "cluster_output"
            cluster_output_dir.mkdir(parents=True, exist_ok=True)
            filepath = cluster_output_dir / "out_0.nxs"

            f = open(filepath, "x")
            f.close()
            js = JobScheduler(working_directory, cluster_output_dir, "b24", "medium.q")

            self.assertFalse(js.timestamp_ok(filepath))


if __name__ == '__main__':
    unittest.main()
