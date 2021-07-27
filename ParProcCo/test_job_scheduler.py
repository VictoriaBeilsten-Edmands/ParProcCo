import drmaa2 as drmaa2
from datetime import datetime
from datetime import timedelta
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from job_scheduler import JobScheduler


def setup_data_files(working_directory, cluster_output_dir):
    # create test files
    input_files = []
    output_files = []
    input_nums = []
    for i in range(4):
        input_file_name = f"data_file_0{i}.txt"
        input_file_path = Path(working_directory) / input_file_name
        input_files.append(input_file_path)
        output_files.append(Path(cluster_output_dir) / f"out_{input_file_name}")

        input_num = str(i**3+3)
        input_nums.append(input_num)

        with open(input_file_path, "w") as f:
            f.write(input_num)

    return input_files, output_files, input_nums


class TestJobScheduler(unittest.TestCase):

    def test_create_template_with_cluster_output_dir(self):
        base_dir = '/dls/tmp/vaq49247/tests/'
        with TemporaryDirectory(prefix='test_dir_', dir=base_dir) as working_directory:
            input_paths = Path('path/to/file.extension')
            cluster_output_dir = os.path.join(working_directory, 'cluster_output_dir')
            with JobScheduler(working_directory, cluster_output_dir, project="b24", priority="medium.q") as js:
                js.create_template(input_paths, "test.sh")
                cluster_output_dir_exists = os.path.exists(cluster_output_dir)
        self.assertTrue(cluster_output_dir_exists, msg="Cluster output directory was not created\n")

    def test_job_scheduler_runs(self):
        # create directory for test files
        base_dir = '/dls/tmp/vaq49247/tests/'
        with TemporaryDirectory(prefix='test_dir_', dir=base_dir) as working_directory:
            cluster_output_dir = Path(working_directory) / "cluster_output"
            script_dir = os.path.join(Path(__file__).absolute().parent.parent, "scripts")
            jobscript = os.path.join(script_dir, "test_script.sh")

            input_files, output_files, input_nums = setup_data_files(working_directory, cluster_output_dir)

            # run jobs
            with JobScheduler(working_directory, cluster_output_dir, "b24", "medium.q") as js:
                js.run(jobscript, input_files)

                # check output files
                for out_file, num in zip(output_files, input_nums):
                    expected_num = str(int(num) * 2)

                    with open(out_file, "r") as f:
                        file_content = f.read()

                    self.assertTrue(os.path.exists(out_file), msg=f"Output file {out_file} was not created\n")
                    self.assertEqual(expected_num, file_content, msg=f"Output file {out_file} content was incorrect\n")

    def test_old_output_timestamps(self):
        # create directory for test files
        base_dir = '/dls/tmp/vaq49247/tests/'
        with TemporaryDirectory(prefix='test_dir_', dir=base_dir) as working_directory:
            cluster_output_dir = Path(working_directory) / "cluster_output"
            script_dir = os.path.join(Path(__file__).absolute().parent.parent, "scripts")
            jobscript = os.path.join(script_dir, "test_script.sh")

            input_paths, _, _ = setup_data_files(working_directory, cluster_output_dir)

            # run jobs
            with JobScheduler(working_directory, cluster_output_dir, "b24", "medium.q") as js:

                js.job_history[js.batch_number] = {}
                js.job_completion_status = {str(input_path): False for input_path in input_paths}
                js.check_jobscript(jobscript)
                js.job_details = []
                js.log_path = None

                session = drmaa2.JobSession()  # Automatically destroyed when it is out of scope
                js.run_jobs(session, jobscript, input_paths)
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

    def test_get_all_queues(self):
        with os.popen('qconf -sql') as q_proc:
            q_name_list = q_proc.read().split()
        ms = drmaa2.MonitoringSession('ms-01')
        qi_list = ms.get_all_queues(q_name_list)
        self.assertEqual(len(qi_list), len(q_name_list))
        for qi in qi_list:
            q_name = qi.name
            self.assertTrue(q_name in q_name_list)

    def test_project_is_none(self):
        base_dir = '/dls/tmp/vaq49247/tests/'
        with TemporaryDirectory(prefix='test_dir_', dir=base_dir) as working_directory:
            cluster_output_dir = Path(working_directory) / "cluster_output"

            with self.assertRaises(ValueError) as context:
                JobScheduler(working_directory, cluster_output_dir, None, "medium.q")
            self.assertTrue("project is empty" in str(context.exception))

    def test_project_is_empty(self):
        base_dir = '/dls/tmp/vaq49247/tests/'
        with TemporaryDirectory(prefix='test_dir_', dir=base_dir) as working_directory:
            cluster_output_dir = Path(working_directory) / "cluster_output"

            with self.assertRaises(ValueError) as context:
                JobScheduler(working_directory, cluster_output_dir, "", "medium.q")
            self.assertTrue("project is empty" in str(context.exception))

    def test_bad_project_name(self):
        base_dir = '/dls/tmp/vaq49247/tests/'
        with TemporaryDirectory(prefix='test_dir_', dir=base_dir) as working_directory:
            cluster_output_dir = Path(working_directory) / "cluster_output"

            with self.assertRaises(ValueError) as context:
                JobScheduler(working_directory, cluster_output_dir, "bad_project_name", "medium.q")
            self.assertTrue("bad_project_name not in list of project names" in str(context.exception))

    def test_queue_is_none(self):
        base_dir = '/dls/tmp/vaq49247/tests/'
        with TemporaryDirectory(prefix='test_dir_', dir=base_dir) as working_directory:
            cluster_output_dir = Path(working_directory) / "cluster_output"

            with self.assertRaises(ValueError) as context:
                JobScheduler(working_directory, cluster_output_dir, "b24", None)
            self.assertTrue("priority is empty" in str(context.exception))

    def test_queue_is_empty(self):
        base_dir = '/dls/tmp/vaq49247/tests/'
        with TemporaryDirectory(prefix='test_dir_', dir=base_dir) as working_directory:
            cluster_output_dir = Path(working_directory) / "cluster_output"

            with self.assertRaises(ValueError) as context:
                JobScheduler(working_directory, cluster_output_dir, "b24", "")
            self.assertTrue("priority is empty" in str(context.exception))

    def test_bad_queue_name(self):
        base_dir = '/dls/tmp/vaq49247/tests/'
        with TemporaryDirectory(prefix='test_dir_', dir=base_dir) as working_directory:
            cluster_output_dir = Path(working_directory) / "cluster_output"

            with self.assertRaises(ValueError) as context:
                JobScheduler(working_directory, cluster_output_dir, "b24", "bad_queue_name.q")
            self.assertTrue("priority bad_queue_name.q not in queue list" in str(context.exception))

    def test_job_times_out(self):
        base_dir = '/dls/tmp/vaq49247/tests/'
        with TemporaryDirectory(prefix='test_dir_', dir=base_dir) as working_directory:
            cluster_output_dir = Path(working_directory) / "cluster_output"
            script_dir = os.path.join(Path(__file__).absolute().parent.parent, "scripts")
            jobscript = os.path.join(script_dir, "test_sleeper_script.sh")

            input_files, _, _ = setup_data_files(working_directory, cluster_output_dir)

            # run jobs
            with JobScheduler(working_directory, cluster_output_dir, "b24", "medium.q",
                              timeout=timedelta(seconds=1)) as js:
                js.run(jobscript, input_files)
                jh = js.job_history
                self.assertEqual(len(jh), 1, f"There should be one batch of jobs; job_history: {jh}\n")
                returned_jobs = jh[0]
                self.assertEqual(len(returned_jobs), 4)
                for job_id in returned_jobs:
                    self.assertEqual(returned_jobs[job_id]["exit_stat"], 137)
                    self.assertEqual(returned_jobs[job_id]["term_sig"], "SIGKILL")
                    self.assertEqual(returned_jobs[job_id]["j_state"], "FAILED")

    def test_bad_jobscript_name(self):
        base_dir = '/dls/tmp/vaq49247/tests/'
        with TemporaryDirectory(prefix='test_dir_', dir=base_dir) as working_directory:
            cluster_output_dir = Path(working_directory) / "cluster_output"

            js = JobScheduler(working_directory, cluster_output_dir, "b24", "medium.q")
            input_files, _, _ = setup_data_files(working_directory, cluster_output_dir)
            script_dir = os.path.join(Path(__file__).absolute().parent.parent, "scripts")
            jobscript = os.path.join(script_dir, "bad_jobscript_name.sh")

            with self.assertRaises(FileNotFoundError) as context:
                js.run(jobscript, input_files)

            self.assertTrue(f"{jobscript} does not exist\n" in str(context.exception))

    def test_insufficient_jobscript_permissions(self):
        base_dir = '/dls/tmp/vaq49247/tests/'
        with TemporaryDirectory(prefix='test_dir_', dir=base_dir) as working_directory:
            cluster_output_dir = Path(working_directory) / "cluster_output"

            js = JobScheduler(working_directory, cluster_output_dir, "b24", "medium.q")
            input_files, _, _ = setup_data_files(working_directory, cluster_output_dir)
            script_dir = os.path.join(Path(__file__).absolute().parent.parent, "scripts")
            jobscript = os.path.join(script_dir, "test_bad_permissions.sh")

            with self.assertRaises(PermissionError) as context:
                js.run(jobscript, input_files)

            self.assertTrue(f"{jobscript} is not executable by user\n" in str(context.exception))


if __name__ == '__main__':
    unittest.main()
