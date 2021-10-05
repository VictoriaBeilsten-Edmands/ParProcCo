from __future__ import annotations

import getpass
import logging
import os.path
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from test_job_controller import setup_jobscript, setup_data_file
from job_controller import get_absolute_path


class TestClusterSubmit(unittest.TestCase):

    def setUp(self) -> None:
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

    def test_rsmap(self) -> None:
        with TemporaryDirectory(prefix='test_dir_', dir=self.base_dir) as working_directory:
            os.chdir(working_directory)
            input_path = "/home/vaq49247/msmapper_test_work/test_dir_0/i07-394487-applied-whole.nxs"
            cluster_output_dir = Path(working_directory) / "cluster_output_dir"
            output_path = cluster_output_dir / "out_i07-394487-applied-whole_0.nxs"

            if not cluster_output_dir.exists():
                logging.debug(f"Making directory {cluster_output_dir}")
                cluster_output_dir.mkdir(exist_ok=True, parents=True)
            script_args = ["rs_map", "-o", str(output_path), "-I", "0::4", "-s", "0.01", input_path]
            proc = subprocess.Popen(script_args)
            proc.communicate()
            print("complete")

    def test_rsmap_prefix(self) -> None:
        with TemporaryDirectory(prefix='test_dir_', dir=self.base_dir) as working_directory:
            os.chdir(working_directory)
            input_path = "/home/vaq49247/msmapper_test_work/test_dir_0/i07-394487-applied-whole.nxs"
            cluster_output_dir = Path(working_directory) / "cluster_output_dir"
            output_path = cluster_output_dir / "out_i07-394487-applied-whole_0.nxs"
            script_path = get_absolute_path("rs_map_prefix")

            if not cluster_output_dir.exists():
                logging.debug(f"Making directory {cluster_output_dir}")
                cluster_output_dir.mkdir(exist_ok=True, parents=True)
            script_args = [script_path, "--input_path", input_path, "--output_path", str(output_path), "-I", "0::4"]
            proc = subprocess.Popen(script_args)
            proc.communicate()
            print("complete")


if __name__ == '__main__':
    unittest.main()
