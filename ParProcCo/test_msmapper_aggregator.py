import getpass
from pathlib import Path
from tempfile import TemporaryDirectory
import logging
import h5py
import numpy as np
import unittest
from typing import List

from msmapper_aggregator import MSMAggregator


def setup_data_files(working_directory: Path) -> List[Path]:
    # create test files
    file_paths = [Path(working_directory) / f"file_0{i}.txt" for i in range(4)]
    file_contents = ["0\n8\n", "2\n10\n", "4\n12\n", "6\n14\n"]
    for file_path, content in zip(file_paths, file_contents):
        with open(file_path, "w") as f:
            f.write(content)
    return file_paths


class TestDataSlicer(unittest.TestCase):

    def setUp(self) -> None:
        current_user = getpass.getuser()
        tmp_dir = f"/dls/tmp/{current_user}/"
        self.base_dir = f"/dls/tmp/{current_user}/tests/"
        self.assertTrue(Path(tmp_dir).is_dir(), f"{tmp_dir} is not a directory")
        if not Path(self.base_dir).is_dir():
            logging.debug(f"Making directory {self.base_dir}")
            Path(self.base_dir).mkdir(exist_ok=True)

    def test_renormalise(self) -> None:
        output_file_paths = ["/scratch/victoria/i07-394487-applied-halfa.nxs",
                             "/scratch/victoria/i07-394487-applied-halfb.nxs"]
        aggregator = MSMAggregator(2)
        aggregator._renormalise(output_file_paths)
        total_volume = aggregator.total_volume
        total_weights = aggregator.accumulator_weights
        with h5py.File("/scratch/victoria/i07-394487-applied-whole.nxs", "r") as f:
            volumes_array = np.array(f["processed"]["reciprocal_space"]["volume"])
            weights_array = np.array(f["processed"]["reciprocal_space"]["weight"])
        np.testing.assert_allclose(total_volume, volumes_array, rtol=0.001)
        np.testing.assert_allclose(total_weights, weights_array, rtol=0.001)

    def test_fill_axes_fields(self) -> None:
        pass

    def test_initialise_accumulator_arrays(self) -> None:
        pass

    def test_accumulate_volumes(self) -> None:
        pass

    def test_write_aggregation_file(self) -> None:
        output_file_paths = ["/scratch/victoria/i07-394487-applied-halfa.nxs",
                             "/scratch/victoria/i07-394487-applied-halfb.nxs"]
        sliced_data_files = [Path(x) for x in output_file_paths]
        with TemporaryDirectory(prefix='test_dir_', dir=self.base_dir) as working_directory:
            cluster_output_dir = Path(working_directory) / "cluster_output"
            if not cluster_output_dir.exists():
                cluster_output_dir.mkdir(exist_ok=True, parents=True)

            aggregator = MSMAggregator(2)
            aggregator_filepath = aggregator.aggregate(cluster_output_dir, sliced_data_files)
            total_volume = aggregator.total_volume
            total_weights = aggregator.accumulator_weights
            with h5py.File("/scratch/victoria/i07-394487-applied-whole.nxs", "r") as f:
                volumes_array = np.array(f["processed"]["reciprocal_space"]["volume"])
                weights_array = np.array(f["processed"]["reciprocal_space"]["weight"])
            np.testing.assert_allclose(total_volume, volumes_array, rtol=0.001)
            np.testing.assert_allclose(total_weights, weights_array, rtol=0.001)

    # def test_get_starts_and_stops(self) -> None:
    #     output_file_paths = ["/scratch/victoria/i07-394487-applied-halfa.nxs",
    #                          "/scratch/victoria/i07-394487-applied-halfb.nxs"]
    #     aggregator = MSMAggregator(2)
    #     aggregator._fill_axes_fields(output_file_paths)
    #
    #     aggregator.hkl_mins = [np.nan, np.nan, np.nan]
    #     aggregator.hkl_maxs = [np.nan, np.nan, np.nan]
    #     aggregator._hkl_axes = []


if __name__ == '__main__':
    unittest.main()
