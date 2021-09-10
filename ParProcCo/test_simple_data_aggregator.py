from __future__ import annotations

import getpass
import logging
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List

from simple_data_aggregator import SimpleDataAggregator


def setup_data_files(working_directory: Path) -> List[Path]:
    # create test files
    file_paths = [working_directory / f"file_0{i}.txt" for i in range(4)]
    file_contents = ["0\n8\n", "2\n10\n", "4\n12\n", "6\n14\n"]
    for file_path, content in zip(file_paths, file_contents):
        with open(file_path, "w") as f:
            f.write(content)
    return file_paths


class TestDataAggregator(unittest.TestCase):

    def setUp(self) -> None:
        current_user = getpass.getuser()
        tmp_dir = f"/dls/tmp/{current_user}/"
        self.base_dir = f"/dls/tmp/{current_user}/tests/"
        self.assertTrue(Path(tmp_dir).is_dir(), f"{tmp_dir} is not a directory")
        if not Path(self.base_dir).is_dir():
            logging.debug(f"Making directory {self.base_dir}")
            Path(self.base_dir).mkdir(exist_ok=True)

    def test_aggregate_data(self) -> None:
        with TemporaryDirectory(prefix='test_dir_', dir=self.base_dir) as working_directory:
            cluster_output_dir = Path(working_directory) / "cluster_output"
            if not cluster_output_dir.exists():
                cluster_output_dir.mkdir(exist_ok=True, parents=True)
            sliced_data_files = setup_data_files(cluster_output_dir)
            written_data = []
            for data_file in sliced_data_files:
                with open(data_file, "r") as f:
                    lines = f.readlines()
                    written_data.append(lines)

            self.assertEqual(written_data, [["0\n", "8\n"], ["2\n", "10\n"], ["4\n", "12\n"], ["6\n", "14\n"]])

            aggregator = SimpleDataAggregator()
            agg_data_path = aggregator.aggregate(4, cluster_output_dir, sliced_data_files)
            with open(agg_data_path, "r") as af:
                agg_data = af.readlines()

            self.assertEqual(agg_data, ["0\n", "8\n", "2\n", "10\n", "4\n", "12\n", "6\n", "14\n"])


if __name__ == '__main__':
    unittest.main()
