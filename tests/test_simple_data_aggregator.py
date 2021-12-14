from __future__ import annotations

import getpass
import logging
import pytest
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from example.simple_data_aggregator import SimpleDataAggregator
from tests.utils import get_tmp_dir_and_workflow, setup_aggregator_data_files


tmp_dir, workflow = get_tmp_dir_and_workflow()


class TestDataAggregator(unittest.TestCase):

    def setUp(self) -> None:
        logging.getLogger().setLevel(logging.INFO)
        self.base_dir = f"{tmp_dir}/tests/"
        self.assertTrue(Path(tmp_dir).is_dir(), f"{tmp_dir} is not a directory")
        if not Path(self.base_dir).is_dir():
            logging.debug(f"Making directory {self.base_dir}")
            Path(self.base_dir).mkdir(exist_ok=True)

    def test_aggregate_data(self) -> None:
        with TemporaryDirectory(prefix='test_dir_', dir=self.base_dir) as working_directory:
            cluster_output_dir = Path(working_directory) / "cluster_output"
            if not cluster_output_dir.is_dir():
                cluster_output_dir.mkdir(exist_ok=True, parents=True)
            aggregation_output = cluster_output_dir / "aggregated_results.txt"
            sliced_data_files = setup_aggregator_data_files(cluster_output_dir)
            written_data = []
            for data_file in sliced_data_files:
                with open(data_file, "r") as f:
                    lines = f.readlines()
                    written_data.append(lines)

            self.assertEqual(written_data, [["0\n", "8\n"], ["2\n", "10\n"], ["4\n", "12\n"], ["6\n", "14\n"]])

            aggregator = SimpleDataAggregator()
            agg_data_path = aggregator.aggregate(aggregation_output, sliced_data_files)
            self.assertEqual(agg_data_path, aggregation_output)
            with open(agg_data_path, "r") as af:
                agg_data = af.readlines()

            self.assertEqual(agg_data, ["0\n", "8\n", "2\n", "10\n", "4\n", "12\n", "6\n", "14\n"])


if __name__ == '__main__':
    unittest.main()
