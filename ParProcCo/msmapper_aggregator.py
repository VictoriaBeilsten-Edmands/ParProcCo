from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple, Union

import h5py
import numpy as np
from job_controller import AggregatorInterface


class MSMAggregator(AggregatorInterface):

    def __init__(self) -> None:
        self._axes_starts_stops: List[List]
        self.accumulator_axis_lengths: List
        self.accumulator_axis_ranges: List
        self.accumulator_volume: np.ndarray
        self.accumulator_weights: np.ndarray
        self.aux_signal_name: Union[None, str]
        self.axes_maxs: List
        self.axes_mins: List
        self.axes_names: List[str]
        self.axes_spacing: List
        self.data_dimensions: int
        self.data_shape: Tuple[int]
        self.nxdata_name: str
        self.nxentry_name: str
        self.output_data_files: List[Path]
        self.renormalisation: bool
        self.signal_name: str
        self.total_slices: int
        self.total_volume: np.ndarray

    def aggregate(self, total_slices: int, aggregation_output_dir: Path, output_data_files: List[Path]) -> Path:
        """Overrides AggregatorInterface.aggregate"""
        self._check_total_slices(total_slices, output_data_files)
        self._renormalise(output_data_files)
        aggregated_data_file = self._write_aggregation_file(aggregation_output_dir, output_data_files)
        return aggregated_data_file

    def _check_total_slices(self, total_slices: int, output_data_files: List[Path]) -> None:
        if type(total_slices) is int:
            self.total_slices = total_slices
        else:
            raise TypeError(f"total_slices is {type(total_slices)}, should be int\n")

        if len(output_data_files) != self.total_slices:
            raise ValueError(
                f"Number of output files {len(output_data_files)} must equal total_slices {self.total_slices}")

    def _renormalise(self, output_data_files: List[Path]) -> None:
        self.output_data_files = output_data_files
        self._get_nxdata()
        self._fill_axes_fields()
        self._initialise_accumulator_arrays()
        self._accumulate_volumes()

        self.total_volume = self.accumulator_volume / self.accumulator_weights
        self.total_volume[np.isnan(self.total_volume)] = 0

    def _get_nxdata(self):
        data_file = self.output_data_files[0]
        with h5py.File(data_file, "r") as root:
            self.nxentry_name = self._get_default_nxgroup(root, b'NXentry')
            nxentry = root[self.nxentry_name]
            self.nxdata_name = self._get_default_nxgroup(nxentry, b'NXdata')
            nxdata = nxentry[self.nxdata_name]
            self._get_default_signals_and_axes(nxdata)

            signal = nxdata[self.signal_name]
            signal_shape = signal.shape
            signal_dimensions = len(signal_shape)
            axes = [nxdata[axis_name] for axis_name in self.axes_names]
            axes_shapes = tuple(axis.shape[0] for axis in axes)
            axes_dimensions = len(axes)
            assert(signal_shape == axes_shapes)
            assert(signal_dimensions == axes_dimensions)
            self.data_dimensions = signal_dimensions
            self.data_shape = signal_shape
            self.axes_spacing = []
            for axis in axes:
                spacing = round((axis[1] - axis[0]), 6)
                self.axes_spacing.append(spacing)

            if self.renormalisation:
                aux_signal = nxdata[self.aux_signal_name]
                assert(len(aux_signal.shape) == self.data_dimensions)
                assert(aux_signal.shape == self.data_shape)

    def _get_default_nxgroup(self, f: Union[h5py.File, h5py.Group], class_name: str) -> str:
        if "default" in f.attrs:
            group_name = f.attrs["default"]
            return group_name

        f_key_list = list(f.keys())
        for key in f_key_list:
            key_attributes = list(f[key].attrs)
            if ("NX_class" in key_attributes) and (f[key].attrs["NX_class"] == class_name):
                group_name = key
                return group_name
        raise ValueError

    def _get_default_signals_and_axes(self, nxdata: h5py.Dataset) -> None:
        if "auxiliary_signals" in nxdata.attrs:
            aux_signal_names = nxdata.attrs["auxiliary_signals"]
            assert(len(aux_signal_names) == 1)
            self.aux_signal_name = aux_signal_names[0]
            self.renormalisation = True
        else:
            self.aux_signal_name = None
            self.renormalisation = False

        if "signal" in nxdata.attrs:
            self.signal_name = nxdata.attrs["signal"]
            if "axes" in nxdata.attrs:
                self.axes_names = nxdata.attrs["axes"]
                # TODO: add logic to create axes if not in file (from integer sequence starting at 0)
                return
        raise KeyError

    def _fill_axes_fields(self) -> None:
        self.axes_mins = [np.nan] * self.data_dimensions
        self.axes_maxs = [np.nan] * self.data_dimensions
        self._axes_starts_stops = [[] for _ in range(self.data_dimensions)]

        for data_file in self.output_data_files:
            with h5py.File(data_file, "r") as f:
                # TODO: check default dataset is the same in all files
                axes = [np.array(f[self.nxentry_name][self.nxdata_name][axis_name]) for axis_name in self.axes_names]
            for j, axis in enumerate(axes):
                self.axes_mins[j] = np.nanmin([np.nanmin(axis), self.axes_mins[j]])
                self.axes_maxs[j] = np.nanmax([np.nanmax(axis), self.axes_maxs[j]])
                self._axes_starts_stops[j].append([axis[0], axis[-1]])

    def _initialise_accumulator_arrays(self) -> None:
        self.accumulator_axis_lengths = []
        self.accumulator_axis_ranges = []

        for i in range(self.data_dimensions):

            length = int((self.axes_maxs[i] - self.axes_mins[i]) / self.axes_spacing[i]) + 1
            self.accumulator_axis_lengths.append(length)
            ranges = [(round((x * self.axes_spacing[i]), 4) + self.axes_mins[i])
                      for x in range(self.accumulator_axis_lengths[i])]
            self.accumulator_axis_ranges.append(ranges)

        self.accumulator_volume = np.zeros(self.accumulator_axis_lengths)
        self.accumulator_weights = np.zeros(self.accumulator_axis_lengths)

    def _accumulate_volumes(self) -> None:
        for data_file in self.output_data_files:
            with h5py.File(data_file, "r") as f:
                # volumes and weights are 3d arrays
                volumes_array = np.array(f[self.nxentry_name][self.nxdata_name][self.signal_name])
                weights_array = np.array(f[self.nxentry_name][self.nxdata_name][self.aux_signal_name])
                axes = [np.array(f[self.nxentry_name][self.nxdata_name][axis_name]) for axis_name in self.axes_names]

            axes_lengths = volumes_array.shape
            starts, stops = self._get_starts_and_stops(axes, axes_lengths)

            if self.renormalisation:
                volumes_array = np.multiply(volumes_array, weights_array)

            self.accumulator_volume[starts[0]:stops[0], starts[1]:stops[1], starts[2]:stops[2]] += volumes_array
            self.accumulator_weights[starts[0]:stops[0], starts[1]:stops[1], starts[2]:stops[2]] += weights_array

    def _get_starts_and_stops(self, axes: List, axes_lengths: List) -> Tuple[List[int], List[int]]:
        starts = []
        stops = []
        for count, value in enumerate(axes):
            start = int((value[0] - self.axes_mins[count]) / self.axes_spacing[count])
            stop = axes_lengths[count] - start
            starts.append(start)
            stops.append(stop)
            if not np.allclose(value, self.accumulator_axis_ranges[count][start:stop]):
                raise RuntimeError
        return starts, stops

    def _write_aggregation_file(self, aggregation_output_dir: Path, output_data_files: List[Path]) -> Path:
        aggregated_data_file = aggregation_output_dir / "aggregated_results.nxs"

        with h5py.File(aggregated_data_file, "w") as f:
            # TODO: change this to use default group names
            processed = f.create_group("processed")
            processed.attrs["NX_class"] = "NXentry"
            processed.attrs["default"] = "reciprocal_space"

            process = f.create_group("processed/process")
            process.attrs["NX_class"] = "NXprocess"
            f.create_dataset("processed/process/date", data=str(datetime.now(timezone.utc)))
            f.create_dataset("processed/process/parameters",
                             data=f"inputs: {output_data_files}, output: {aggregated_data_file}")
            f.create_dataset("processed/process/program", data="ParProcCo")
            f.create_dataset("processed/process/version", data="1.0")

            reciprocal_space = f.create_group("processed/reciprocal_space")
            reciprocal_space.attrs["NX_class"] = "NXdata"
            reciprocal_space.attrs["auxillary_signals"] = "weight"
            reciprocal_space.attrs["axes"] = ["h-axis", "k-axis", "l-axis"]
            reciprocal_space.attrs["signals"] = "volume"
            for i, axis in enumerate(["h", "k", "l"]):
                reciprocal_space.attrs[f"{axis}-axis_indices"] = i
                f.create_dataset(f"processed/reciprocal_space/{axis}-axis", data=self.accumulator_axis_ranges[i])
            f.create_dataset("processed/reciprocal_space/volume", data=self.total_volume)
            f.create_dataset("processed/reciprocal_space/weight", data=self.accumulator_weights)

            f.attrs["default"] = "processed"

            for i, filepath in enumerate(output_data_files):
                f[f"entry{i}"] = h5py.ExternalLink(str(filepath), "/processed")

            binoculars = f.create_group("binoculars")
            binoculars.attrs["type"] = "space"

            f.create_group("binoculars/axes")
            for i, axis in enumerate(["H", "K", "L"]):
                axis_min = self.axes_mins[i]
                axis_max = self.axes_maxs[i]
                scaling = (self.accumulator_axis_lengths[i] - 1) / (axis_max - axis_min)
                axis_dataset = [i, axis_min, axis_max, self.axes_spacing[i], axis_min * scaling, axis_max * scaling]
                f.create_dataset(f"binoculars/axes/{axis}", data=axis_dataset)

            binoculars["counts"] = f["processed/reciprocal_space/volume"]
            binoculars["contributions"] = f["processed/reciprocal_space/weight"]

        return aggregated_data_file
