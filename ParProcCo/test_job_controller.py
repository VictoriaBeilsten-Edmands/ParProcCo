from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
import getpass
import logging
import os
import unittest

from job_controller import JobController
from simple_data_slicer import SimpleDataSlicer
from simple_data_aggregator import SimpleDataAggregator


def setup_data_file(working_directory):
    # create test files
    file_name = "test_raw_data.txt"
    input_file_path = Path(working_directory) / file_name
    with open(input_file_path, "w") as f:
        f.write("0\n1\n2\n3\n4\n5\n6\n7\n")
    return input_file_path


def setup_jobscript(working_directory):
    jobscript = Path(working_directory) / "test_script.py"
    with open(jobscript, "x") as f:
        jobscript_lines = """
#!/usr/bin/env python3

import argparse
import sys


def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", help="str: path to input file", type=str)
    parser.add_argument("--output_path", help="str: path to output file", type=str)
    parser.add_argument("--start", help="index to start at (inclusive)")
    parser.add_argument("--stop", help="index to end at (exclusive)")
    parser.add_argument("--step", help="step size")

    return parser

def check_args(args):
    empty_fields = [k for k, v in vars(args).items() if v is None]
    if len(empty_fields) > 0:
        raise ValueError(f"Missing arguments: {empty_fields}")

def write_lines(input_path, output_path, start, stop, step):
    with open(input_path, "r") as in_f:
        for i, line in enumerate(in_f):
            if i >= stop:
                break

            elif i >= start and ((i - start) % step == 0):
                doubled = int(line.strip("\\n"))*2
                doubled_str = f"{doubled}\\n"
                with open(output_path, "a+") as out_f:
                    out_f.write(doubled_str)

if __name__ == '__main__':
    '''
    $ python jobscript.py --input_path input_path --output_path output_path --start start --stop stop --step step
    '''
    parser = setup_parser()
    args = parser.parse_args()
    check_args(args)

    write_lines(args.input_path, args.output_path, int(args.start), int(args.stop), int(args.step))
"""
        jobscript_lines = jobscript_lines.lstrip()
        f.write(jobscript_lines)
    os.chmod(jobscript, 0o777)
    return jobscript


class TestJobController(unittest.TestCase):

    def setUp(self):
        current_user = getpass.getuser()
        tmp_dir = f"/dls/tmp/{current_user}/"
        self.base_dir = f"/dls/tmp/{current_user}/tests/"
        self.assertTrue(Path(tmp_dir).is_dir(), f"{tmp_dir} is not a directory")
        if not Path(self.base_dir).is_dir():
            logging.debug(f"Making directory {self.base_dir}")
            Path(self.base_dir).mkdir(exist_ok=True)

    def test_end_to_end(self):
        with TemporaryDirectory(prefix='test_dir_', dir=self.base_dir) as working_directory:
            cluster_output_dir = Path(working_directory) / "cluster_output"

            jobscript = setup_jobscript(working_directory)

            input_file_path = setup_data_file(working_directory)

            jc = JobController(working_directory, cluster_output_dir, project="b24", priority="medium.q")
            agg_data_path = jc.run(SimpleDataSlicer(), SimpleDataAggregator(4), input_file_path, 4, jobscript)

            self.assertEqual(agg_data_path, Path(cluster_output_dir) / "aggregated_results.txt")
            with open(agg_data_path, "r") as af:
                agg_data = af.readlines()

            self.assertEqual(agg_data, ["0\n", "8\n", "2\n", "10\n", "4\n", "12\n", "6\n", "14\n"])

    def test_all_jobs_fail(self):
        with TemporaryDirectory(prefix='test_dir_', dir=self.base_dir) as working_directory:
            cluster_output_dir = Path(working_directory) / "cluster_output"

            jobscript = setup_jobscript(working_directory)
            with open(jobscript, "a+") as f:
                f.write("import time\ntime.sleep(5)\n")

            input_file_path = setup_data_file(working_directory)

            jc = JobController(working_directory, cluster_output_dir, project="b24", priority="medium.q",
                               timeout=timedelta(seconds=1))
            with self.assertRaises(RuntimeError) as context:
                jc.run(SimpleDataSlicer(), SimpleDataAggregator(4), input_file_path, 4, jobscript)
            self.assertTrue(f"All jobs failed\n" in str(context.exception))


if __name__ == '__main__':
    unittest.main()
