from __future__ import annotations

import getpass
import logging
import os
import unittest
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

from example.simple_wrapper import SimpleWrapper
from ParProcCo.job_controller import JobController
from tests.utils import setup_aggregation_script, setup_data_file, setup_runner_script, setup_jobscript

from tests.test_job_scheduler import CLUSTER_PROJ, CLUSTER_QUEUE, CLUSTER_RESOURCES

class TestJobController(unittest.TestCase):

    def setUp(self) -> None:
        logging.getLogger().setLevel(logging.INFO)
        current_user = getpass.getuser()
        tmp_dir = f"/dls/tmp/{current_user}/"
        self.base_dir = f"/dls/tmp/{current_user}/tests/"
        self.assertTrue(Path(tmp_dir).is_dir(), f"{tmp_dir} is not a directory")
        if not Path(self.base_dir).is_dir():
            logging.debug(f"Making directory {self.base_dir}")
            Path(self.base_dir).mkdir(exist_ok=True)
        self.current_dir = os.getcwd()

    def tearDown(self):
        os.chdir(self.current_dir)

    def test_all_jobs_fail(self) -> None:
        with TemporaryDirectory(prefix='test_dir_', dir=self.base_dir) as working_directory:
            os.chdir(working_directory)
            cluster_output_name = "cluster_output"
            os.mkdir(cluster_output_name, 0o775)

            runner_script = setup_runner_script(working_directory)
            jobscript = setup_jobscript(working_directory)
            aggregation_script = setup_aggregation_script(working_directory)
            with open(jobscript, "a+") as f:
                f.write("import time\ntime.sleep(60)\n")

            input_path = setup_data_file(working_directory)
            runner_script_args = [str(jobscript), "--input-path", str(input_path)]

            wrapper = SimpleWrapper(runner_script, aggregation_script)
            wrapper.set_cores(6)
            jc = JobController(wrapper, Path(cluster_output_name), project=CLUSTER_PROJ, queue=CLUSTER_QUEUE, cluster_resources=CLUSTER_RESOURCES,
                                timeout=timedelta(seconds=1))
            with self.assertRaises(RuntimeError) as context:
                jc.run(4, jobscript_args=runner_script_args)
            self.assertTrue(f"All jobs failed. job_history: " in str(context.exception))

    def test_end_to_end(self) -> None:
        with TemporaryDirectory(prefix='test_dir_', dir=self.base_dir) as working_directory:
            os.chdir(working_directory)
            cluster_output_name = "cluster_output"
            os.mkdir(cluster_output_name, 0o775)

            runner_script = setup_runner_script(working_directory)
            jobscript = setup_jobscript(working_directory)
            aggregation_script = setup_aggregation_script(working_directory)

            input_path = setup_data_file(working_directory)
            runner_script_args = [str(jobscript), "--input-path", str(input_path)]
            sliced_results = [Path(working_directory) / f"out_{i}" for i in range(4)]

            wrapper = SimpleWrapper(runner_script, aggregation_script)
            wrapper.set_cores(6)
            jc = JobController(wrapper, Path(cluster_output_name), project=CLUSTER_PROJ, queue=CLUSTER_QUEUE,
                               cluster_resources=CLUSTER_RESOURCES)
            jc.run(4, jobscript_args=runner_script_args)

            with open(Path(working_directory) / cluster_output_name / "aggregated_results.txt", "r") as af:
                agg_data = af.readlines()

            self.assertEqual(agg_data, ["0\n", "8\n", "2\n", "10\n", "4\n", "12\n", "6\n", "14\n"])
            for result in sliced_results:
                self.assertFalse(result.is_file())

    def test_single_job_does_not_aggregate(self) -> None:
        with TemporaryDirectory(prefix='test_dir_', dir=self.base_dir) as working_directory:
            os.chdir(working_directory)
            cluster_output_name = "cluster_output"
            os.mkdir(cluster_output_name, 0o775)

            runner_script = setup_runner_script(working_directory)
            jobscript = setup_jobscript(working_directory)
            aggregation_script = setup_aggregation_script(working_directory)

            input_path = setup_data_file(working_directory)
            runner_script_args = [str(jobscript), "--input-path", str(input_path)]
            sliced_result = Path(working_directory) / cluster_output_name / f"out_0"
            aggregated_file = Path(working_directory) / cluster_output_name / "aggregated_results.txt"

            wrapper = SimpleWrapper(runner_script, aggregation_script)
            wrapper.set_cores(6)
            jc = JobController(wrapper, Path(cluster_output_name), project=CLUSTER_PROJ, queue=CLUSTER_QUEUE,
                               cluster_resources=CLUSTER_RESOURCES)
            jc.run(1, jobscript_args=runner_script_args)

            self.assertEqual([sliced_result], jc.sliced_results)
            self.assertFalse(aggregated_file.is_file())
            self.assertTrue(sliced_result.is_file())
            with open(sliced_result, "r") as af:
                agg_data = af.readlines()

            self.assertEqual(agg_data, ["0\n", "2\n", "4\n", "6\n", "8\n", "10\n", "12\n", "14\n"])


if __name__ == '__main__':
    unittest.main()
