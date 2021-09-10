from __future__ import annotations

import getpass
import logging
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import List

import h5py
import numpy as np
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
        output_file_paths = [Path("/scratch/victoria/i07-394487-applied-halfa.nxs"),
                             Path("/scratch/victoria/i07-394487-applied-halfb.nxs")]
        aggregator = MSMAggregator()
        aggregator._check_total_slices(2, output_file_paths)
        aggregator._renormalise(output_file_paths)
        total_volume = aggregator.total_volume
        total_weights = aggregator.accumulator_weights
        with h5py.File("/scratch/victoria/i07-394487-applied-whole.nxs", "r") as f:
            volumes_array = np.array(f["processed"]["reciprocal_space"]["volume"])
            weights_array = np.array(f["processed"]["reciprocal_space"]["weight"])
        np.testing.assert_allclose(total_volume, volumes_array, rtol=0.001)
        np.testing.assert_allclose(total_weights, weights_array, rtol=0.001)

    def test_check_total_slices_not_int(self) -> None:
        total_slices = "1"
        output_data_files = [Path("file/path/a"), Path("file/path/b")]
        aggregator = MSMAggregator()

        with self.assertRaises(TypeError) as context:
            aggregator._check_total_slices(total_slices, output_data_files)
        self.assertTrue("total_slices is <class 'str'>, should be int" in str(context.exception))
        self.assertRaises(AttributeError, lambda: aggregator.total_slices)

    def test_check_total_slices_length_wrong(self) -> None:
        total_slices = 2
        output_data_files = [Path("file/path/a")]
        aggregator = MSMAggregator()

        with self.assertRaises(ValueError) as context:
            aggregator._check_total_slices(total_slices, output_data_files)
        self.assertTrue("Number of output files 1 must equal total_slices 2" in str(context.exception))
        self.assertEqual(total_slices, aggregator.total_slices)

    def test_check_total_slices(self) -> None:
        total_slices = 2
        output_data_files = [Path("file/path/a"), Path("file/path/b")]
        aggregator = MSMAggregator()
        aggregator._check_total_slices(total_slices, output_data_files)
        self.assertEqual(total_slices, aggregator.total_slices)

    # TODO add test for when renormalisation is False and when dimensions are wrong
    def test_get_nxdata(self) -> None:
        output_data_files = ["/scratch/victoria/i07-394487-applied-halfa.nxs",
                             "/scratch/victoria/i07-394487-applied-halfb.nxs"]
        aggregator = MSMAggregator()
        aggregator.output_data_files = output_data_files
        aggregator._get_nxdata()
        self.assertEqual(aggregator.nxentry_name, "processed")
        self.assertEqual(aggregator.nxdata_name, "reciprocal_space")
        self.assertEqual(aggregator.aux_signal_names, ["weight"])
        # TODO: change so only accumulate if other aux_signals
        self.assertEqual(aggregator.accumulate_aux_signals, True)
        self.assertEqual(aggregator.nxentry_name, "processed")
        self.assertEqual(aggregator.renormalisation, True)
        self.assertEqual(aggregator.signal_name, "volume")
        self.assertEqual(aggregator.axes_names, ["h-axis", "k-axis", "l-axis"])
        self.assertEqual(aggregator.data_dimensions, 3)
        self.assertEqual(aggregator.data_shape, (83, 76, 13))
        self.assertEqual(aggregator.axes_spacing, [0.02, 0.02, 0.02])

    # TODO add tests for when no default
    def test_get_default_nxentry(self) -> None:

        data_file = "/scratch/victoria/i07-394487-applied-halfa.nxs"
        aggregator = MSMAggregator()
        with h5py.File(data_file, "r") as f:
            nxentry_name = aggregator._get_default_nxgroup(f, b'NXentry').decode()

        self.assertEqual(nxentry_name, "processed")

    def test_get_default_nxdata(self) -> None:

        data_file = "/scratch/victoria/i07-394487-applied-halfa.nxs"
        aggregator = MSMAggregator()
        with h5py.File(data_file, "r") as f:
            nxdata_name = aggregator._get_default_nxgroup(f["processed"], b'NXdata').decode()

        self.assertEqual(nxdata_name, "reciprocal_space")

    def test_get_default_signals_and_axes(self) -> None:
        data_file = "/scratch/victoria/i07-394487-applied-halfa.nxs"
        aggregator = MSMAggregator()
        with h5py.File(data_file, "r") as f:
            nx_data = f["processed/reciprocal_space"]
            aggregator._get_default_signals_and_axes(nx_data)
        self.assertEqual(aggregator.aux_signal_names, ["weight"])
        self.assertEqual(aggregator.accumulate_aux_signals, True)
        self.assertEqual(aggregator.renormalisation, True)
        self.assertEqual(aggregator.signal_name, "volume")
        self.assertEqual(aggregator.axes_names, ["h-axis", "k-axis", "l-axis"])

    def test_fill_axes_fields(self) -> None:
        aggregator = MSMAggregator()
        aggregator.data_dimensions = 3
        aggregator.output_data_files = ["/scratch/victoria/i07-394487-applied-halfa.nxs",
                             "/scratch/victoria/i07-394487-applied-halfb.nxs"]
        aggregator.nxentry_name = "processed"
        aggregator.nxdata_name = "reciprocal_space"
        aggregator.axes_names = ["h-axis", "k-axis", "l-axis"]
        aggregator._fill_axes_fields()
        self.assertEqual(aggregator.axes_mins, [-0.2, -0.08, 0.86])
        self.assertEqual(aggregator.axes_maxs, [1.4400000000000002, 1.44, 1.1])
        self.assertEqual(aggregator._axes_starts_stops, [[[-0.2, 1.4400000000000002], [-0.2, 1.4400000000000002]],
                                                         [[-0.08, 1.42], [-0.08, 1.44]], [[0.86, 1.1], [0.86, 1.1]]])

    def test_initialise_accumulator_arrays(self) -> None:
        aggregator = MSMAggregator()
        aggregator.data_dimensions = 3
        aggregator.axes_mins = [-0.2, -0.08, 0.86]
        aggregator.axes_maxs = [1.4400000000000002, 1.44, 1.1]
        aggregator.axes_spacing = [0.02, 0.02, 0.02]
        aggregator.renormalisation = True

        aggregator._initialise_accumulator_arrays()
        self.assertEqual(aggregator.accumulator_axis_lengths, [83, 77, 13])
        self.assertEqual(len(aggregator.accumulator_axis_ranges), 3)
        self.assertEqual(len(aggregator.accumulator_axis_ranges[0]), 83)
        self.assertEqual(len(aggregator.accumulator_axis_ranges[1]), 77)
        self.assertEqual(len(aggregator.accumulator_axis_ranges[2]), 13)
        self.assertEqual(aggregator.accumulator_axis_ranges[0][0], -0.2)
        self.assertEqual(aggregator.accumulator_axis_ranges[0][-1], 1.44)
        self.assertEqual(aggregator.accumulator_axis_ranges[1][0], -0.08)
        self.assertEqual(aggregator.accumulator_axis_ranges[1][-1], 1.44)
        self.assertEqual(aggregator.accumulator_axis_ranges[2][0], 0.86)
        self.assertEqual(aggregator.accumulator_axis_ranges[2][-1], 1.1)
        self.assertTrue(np.array_equal(aggregator.accumulator_volume, np.zeros([83, 77, 13])))
        self.assertTrue(np.array_equal(aggregator.accumulator_weights, np.zeros([83, 77, 13])))

    def test_accumulate_volumes(self) -> None:
        aggregator = MSMAggregator()
        aggregator.output_data_files = ["/scratch/victoria/i07-394487-applied-halfa.nxs",
                             "/scratch/victoria/i07-394487-applied-halfb.nxs"]
        aggregator.nxentry_name = "processed"
        aggregator.nxdata_name = "reciprocal_space"
        aggregator.signal_name = "volume"
        aggregator.axes_names = ["h-axis", "k-axis", "l-axis"]
        aggregator.renormalisation = True
        aggregator.data_dimensions = 3
        aggregator.accumulator_weights = np.zeros([83, 77, 13])
        aggregator.accumulator_volume = np.zeros([83, 77, 13])
        aggregator.axes_mins = [-0.2, -0.08, 0.86]
        aggregator.axes_spacing = [0.02, 0.02, 0.02]
        aggregator.accumulator_axis_lengths = [83, 77, 13]
        aggregator.accumulator_axis_ranges = [[(round((x * aggregator.axes_spacing[i]), 4) + aggregator.axes_mins[i])
                                               for x in range(aggregator.accumulator_axis_lengths[i])]
                                              for i in range(3)]

        aggregator._accumulate_volumes()

        with h5py.File("/scratch/victoria/i07-394487-applied-whole.nxs", "r") as f:
            volumes_array = np.array(f["processed"]["reciprocal_space"]["volume"])
            weights_array = np.array(f["processed"]["reciprocal_space"]["weight"])
        self.assertEqual(aggregator.total_volume.shape, (83, 77, 13))
        np.testing.assert_allclose(aggregator.total_volume, volumes_array, rtol=0.001)
        np.testing.assert_allclose(aggregator.accumulator_weights, weights_array, rtol=0.001)

    def test_get_starts_and_stops(self) -> None:
        aggregator = MSMAggregator()
        aggregator.axes_mins = [-0.2, -0.08, 0.86]
        aggregator.axes_spacing = [0.02, 0.02, 0.02]
        aggregator.accumulator_axis_lengths = [83, 77, 13]
        aggregator.accumulator_axis_ranges = [[(round((x * aggregator.axes_spacing[i]), 4) + aggregator.axes_mins[i])
                                               for x in range(aggregator.accumulator_axis_lengths[i])]
                                              for i in range(3)]
        aggregator.data_dimensions = 3
        data_file = "/scratch/victoria/i07-394487-applied-halfa.nxs"
        with h5py.File(data_file, "r") as f:
            axes = [np.array(f["processed/reciprocal_space"][axis_name]) for axis_name in ["h-axis", "k-axis", "l-axis"]]
        slices = aggregator._get_starts_and_stops(axes, [83, 76, 13])
        self.assertEqual(slices, [slice(0, 83), slice(0, 76), slice(0, 13)])


    def test_write_aggregation_file(self) -> None:
        output_file_paths = ["/scratch/victoria/i07-394487-applied-halfa.nxs",
                             "/scratch/victoria/i07-394487-applied-halfb.nxs"]
        sliced_data_files = [Path(x) for x in output_file_paths]
        with TemporaryDirectory(prefix='test_dir_', dir=self.base_dir) as working_directory:
            cluster_output_dir = Path(working_directory) / "cluster_output"
            if not cluster_output_dir.exists():
                cluster_output_dir.mkdir(exist_ok=True, parents=True)

            aggregator = MSMAggregator()
            aggregator_filepath = aggregator.aggregate(2, cluster_output_dir, sliced_data_files)
            total_volume = aggregator.total_volume
            total_weights = aggregator.accumulator_weights
            with h5py.File("/scratch/victoria/i07-394487-applied-whole.nxs", "r") as f:
                volumes_array = np.array(f["processed"]["reciprocal_space"]["volume"])
                weights_array = np.array(f["processed"]["reciprocal_space"]["weight"])
            np.testing.assert_allclose(total_volume, volumes_array, rtol=0.001)
            np.testing.assert_allclose(total_weights, weights_array, rtol=0.001)

            self.assertEqual(aggregator_filepath, cluster_output_dir / "aggregated_results.nxs")
            with h5py.File(aggregator_filepath, "r") as af:
                aggregated_volumes = np.array(af["processed"]["reciprocal_space"]["volume"])
                aggregated_weights = np.array(af["processed"]["reciprocal_space"]["weight"])
            np.testing.assert_allclose(volumes_array, aggregated_volumes, rtol=0.001)
            np.testing.assert_allclose(weights_array, aggregated_weights, rtol=0.001)


if __name__ == '__main__':
    unittest.main()
