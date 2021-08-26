import h5py
from datetime import datetime, timezone
import numpy as np
from pathlib import Path
from typing import List, Tuple


class MSMAggregator:

    def __init__(self, total_slices: int) -> None:

        self.output_data_files: List[str]
        self.hkl_spacing: List
        self.accumulator_hkl_lengths: List
        self.accumulator_hkl_ranges: List
        self._hkl_axes: List[List[np.ndarray]]
        self.accumulator_volume: np.ndarray
        self.accumulator_weights: np.ndarray
        self.total_volume = np.ndarray

        if type(total_slices) is int:
            self.total_slices = total_slices
        else:
            raise TypeError(f"total_slices is {type(total_slices)}, should be int\n")

    def aggregate(self, aggregation_output_dir: Path, output_data_files: List[Path]) -> Path:
        self._renormalise(output_data_files)
        aggregated_data_file = self._write_aggregation_file(aggregation_output_dir, output_data_files)
        return aggregated_data_file

    def _renormalise(self, output_data_files: List[str]) -> None:
        self.output_data_files = output_data_files
        self._fill_axes_fields()
        self._initialise_accumulator_arrays()
        self._accumulate_volumes()

        self.total_volume = self.accumulator_volume / self.accumulator_weights
        self.total_volume[np.isnan(self.total_volume)] = 0

    def _write_aggregation_file(self, aggregation_output_dir: Path, output_data_files: List[Path]) -> Path:
        aggregated_data_file = Path(aggregation_output_dir) / "aggregated_results.nxs"

        with h5py.File(aggregated_data_file, "w") as f:
            binoculars = f.create_group("binoculars")
            binoculars.attrs["type"] = "space"
            f.create_dataset("binoculars/counts", data=self.total_volume)
            f.create_dataset("binoculars/contributions", data=self.accumulator_weights)
            f.create_group("binoculars/axes")
            for i, axis in enumerate(["H", "K", "L"]):
                axis_min = self.hkl_mins[i]
                axis_max = self.hkl_maxs[i]
                scaling = (self.accumulator_hkl_lengths[i] - 1) / (axis_max - axis_min)
                axis_dataset = [i, axis_min, axis_max, self.hkl_spacing[i], axis_min * scaling, axis_max * scaling]
                f.create_dataset(f"binoculars/axes/{axis}", data=axis_dataset)

            for i, filepath in enumerate(output_data_files):
                f[f"entry{i}"] = h5py.ExternalLink(str(filepath), "/processed")

            processed = f.create_group("processed")
            processed.attrs["NX_class"] = "NXentry"
            processed.attrs["default"] = "reciprocal_space"

            process = f.create_group("processed/process")
            process.attrs["NX_class"] = "NXprocess"
            f.create_dataset("processed/process/date", data=str(datetime.now(timezone.utc)))
            f.create_dataset("processed/process/parameters", data=f"inputs: {output_data_files}, output: {aggregated_data_file}")
            f.create_dataset("processed/process/program", data="ParProcCo")
            f.create_dataset("processed/process/version", data="1.0")

            reciprocal_space = f.create_group("processed/reciprocal_space")
            reciprocal_space.attrs["NX_class"] = "NXdata"
            reciprocal_space.attrs["auxillary_signals"] = "weight"
            reciprocal_space.attrs["axes"] = ["h-axis", "k-axis", "l-axis"]
            reciprocal_space.attrs["signals"] = "volume"
            for i, axis in enumerate(["h", "k", "l"]):
                reciprocal_space.attrs[f"{axis}-axis_indices"] = i
                f.create_dataset(f"processed/reciprocal_space/{axis}-axis", data=self.accumulator_hkl_ranges[i])
            f.create_dataset("processed/reciprocal_space/volume", data=f["binoculars/counts"])
            f.create_dataset("processed/reciprocal_space/weight", data=f["binoculars/contributions"])

            f.attrs["default"] = "processed"

        return aggregated_data_file

    def _fill_axes_fields(self) -> None:
        self.hkl_mins = [np.nan, np.nan, np.nan]
        self.hkl_maxs = [np.nan, np.nan, np.nan]
        self._hkl_axes = []

        for data_file in self.output_data_files:
            with h5py.File(data_file, "r") as f:
                hkl_axes = [np.array(f["processed"]["reciprocal_space"]["h-axis"]),
                            np.array(f["processed"]["reciprocal_space"]["k-axis"]),
                            np.array(f["processed"]["reciprocal_space"]["l-axis"])]
            self._hkl_axes.append(hkl_axes)
            for i, axis in enumerate(hkl_axes):
                self.hkl_mins[i] = np.nanmin([np.nanmin(axis), self.hkl_mins[i]])
                self.hkl_maxs[i] = np.nanmax([np.nanmax(axis), self.hkl_maxs[i]])

    def _initialise_accumulator_arrays(self) -> None:
        self.hkl_spacing = []
        self.accumulator_hkl_lengths = []
        self.accumulator_hkl_ranges = []

        for i, axis in enumerate(self._hkl_axes[0]):
            spacing = round((axis[1] - axis[0]), 6)
            self.hkl_spacing.append(spacing)
            length = int((self.hkl_maxs[i] - self.hkl_mins[i]) / self.hkl_spacing[i]) + 1
            self.accumulator_hkl_lengths.append(length)
            ranges = [(round((x * self.hkl_spacing[i]), 4) + self.hkl_mins[i]) for x in range(self.accumulator_hkl_lengths[i])]
            self.accumulator_hkl_ranges.append(ranges)

        self.accumulator_volume = np.zeros(self.accumulator_hkl_lengths)
        self.accumulator_weights = np.zeros(self.accumulator_hkl_lengths)

    def _accumulate_volumes(self) -> None:
        for i, data_file in enumerate(self.output_data_files):
            with h5py.File(data_file, "r") as f:
                # volumes and weights are 3d arrays
                volumes_array = np.array(f["processed"]["reciprocal_space"]["volume"])
                weights_array = np.array(f["processed"]["reciprocal_space"]["weight"])
                raw_volumes = np.multiply(volumes_array, weights_array)
                hkl_lengths = volumes_array.shape
                hkl_axes = self._hkl_axes[i]

            starts, stops = self._get_starts_and_stops(hkl_axes, hkl_lengths)

            self.accumulator_volume[starts[0]:stops[0], starts[1]:stops[1], starts[2]:stops[2]] += raw_volumes
            self.accumulator_weights[starts[0]:stops[0], starts[1]:stops[1], starts[2]:stops[2]] += weights_array

    def _get_starts_and_stops(self, hkl_axes: List, hkl_lengths: List) -> Tuple[List[int], List[int]]:
        starts = []
        stops = []
        for count, value in enumerate(hkl_axes):
            start = int((value[0] - self.hkl_mins[count]) / self.hkl_spacing[count])
            stop = hkl_lengths[count] - start
            starts.append(start)
            stops.append(stop)
            if not np.allclose(value, self.accumulator_hkl_ranges[count][start:stop]):
                raise RuntimeError
        return starts, stops
