from __future__ import annotations

import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import AnyStr, List, Tuple, Union

import h5py
import numpy as np
from job_controller import AggregatorInterface

from ParProcCo import __version__


def decode_to_string(any_string: AnyStr) -> str:
    output = any_string.decode() if not isinstance(any_string, str) else any_string
    return output


class MSMAggregator(AggregatorInterface):

    def __init__(self) -> None:
        self.accumulate_aux_signals: bool
        self.accumulator_axis_lengths: List
        self.accumulator_axis_ranges: List
        self.accumulator_volume: np.ndarray
        self.accumulator_weights: np.ndarray
        self.all_slices: List[List[slice]]
        self.aux_signal_names: Union[None, List[str]]
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
        self._initialise_arrays()
        self._accumulate_volumes()

    def _initialise_arrays(self) -> None:
        self.axes_mins = [np.inf] * self.data_dimensions
        self.axes_maxs = [np.NINF] * self.data_dimensions
        self.all_slices = []

        data_group_name = "/".join([self.nxentry_name, self.nxdata_name])
        for data_file in self.output_data_files:
            with h5py.File(data_file, "r") as f:
                axes = [np.array(f[data_group_name][axis_name]) for axis_name in self.axes_names]
            for j, axis in enumerate(axes):
                self.axes_mins[j] = min([min(axis), self.axes_mins[j]])
                self.axes_maxs[j] = max([max(axis), self.axes_maxs[j]])

        for data_file in self.output_data_files:
            with h5py.File(data_file, "r") as f:
                volumes_array = np.array(f[data_group_name][self.signal_name])
                axes = [np.array(f[data_group_name][axis_name]) for axis_name in self.axes_names]

            axes_lengths = tuple(len(axis) for axis in axes)
            assert axes_lengths == volumes_array.shape, "axes_lengths must equal volumes_array.shape"
            slices = []
            for j, axis in enumerate(axes):
                start = int(round((axis[0] - self.axes_mins[j]) / self.axes_spacing[j]))
                stop = axes_lengths[j] + start
                slices.append(slice(start, stop))

            assert len(slices) == self.data_dimensions, "number of slices must equal self.data_dimensions"
            self.all_slices.append(slices)

        self.accumulator_axis_lengths = []
        self.accumulator_axis_ranges = []

        for i in range(self.data_dimensions):
            length = int(round((self.axes_maxs[i] - self.axes_mins[i]) / self.axes_spacing[i])) + 1
            self.accumulator_axis_lengths.append(length)
            ranges = [round(x * self.axes_spacing[i] + self.axes_mins[i], 4)
                      for x in range(self.accumulator_axis_lengths[i])]
            self.accumulator_axis_ranges.append(ranges)

        for i, data_file in enumerate(self.output_data_files):
            with h5py.File(data_file, "r") as f:
                axes = [np.array(f[data_group_name][axis_name]) for axis_name in self.axes_names]
            for j, axis in enumerate(axes):
                if not np.allclose(axis, self.accumulator_axis_ranges[j][self.all_slices[i][j]]):
                    raise RuntimeError(f"axis {j} does not match slice {slice} of accumulator_axis_range")

        self.accumulator_volume = np.zeros(self.accumulator_axis_lengths)
        if self.renormalisation:
            self.accumulator_weights = np.zeros(self.accumulator_axis_lengths)

    def _get_nxdata(self):
        data_file = self.output_data_files[0]
        with h5py.File(data_file, "r") as root:
            self.nxentry_name = self._get_default_nxgroup(root, "NXentry")
            nxentry = root[self.nxentry_name]
            self.nxdata_name = self._get_default_nxgroup(nxentry, "NXdata")
            nxdata = nxentry[self.nxdata_name]
            self._get_default_signals_and_axes(nxdata)

            signal = nxdata[self.signal_name]
            signal_shape = signal.shape
            signal_dimensions = len(signal_shape)
            axes = [nxdata[axis_name] for axis_name in self.axes_names]
            axes_shapes = tuple(axis.shape[0] for axis in axes)
            axes_dimensions = len(axes)
            assert signal_dimensions == axes_dimensions, "signal and axes dimensions must match"
            assert signal_shape == axes_shapes, "signal and axes shapes must match"
            self.data_dimensions = signal_dimensions
            self.data_shape = signal_shape
            self.axes_spacing = [round((axis[1] - axis[0]), 6) for axis in axes]

            if self.renormalisation:
                weights = nxdata["weight"]
                assert len(weights.shape) == self.data_dimensions, "signal and weight dimensions must match"
                assert weights.shape == self.data_shape, "signal and weight shapes must match"

    def _get_default_nxgroup(self, f: Union[h5py.File, h5py.Group], class_name: str) -> str:
        if "default" in f.attrs:
            group_name = f.attrs["default"]
            group_name = decode_to_string(group_name)
            class_type = f[group_name].attrs.get("NX_class", '')
            class_type = decode_to_string(class_type)
            assert class_type == class_name, f"{group_name} class_name must be {class_name}"
            return group_name

        for group_name in f.keys():
            try:
                class_type = f[group_name].attrs.get("NX_class", '')
                class_type = decode_to_string(class_type)
                if class_type == class_name:
                    group_name = decode_to_string(group_name)
                    return group_name
            except KeyError:
                warnings.warn(f"KeyError: {group_name} could not be accessed in {f}")
        raise ValueError(f"no {class_name} group found")

    def _get_default_signals_and_axes(self, nxdata: h5py.Dataset) -> None:
        self.renormalisation = False
        self.accumulate_aux_signals = False

        if "auxiliary_signals" in nxdata.attrs:
            self.aux_signal_names = [decode_to_string(name) for name in nxdata.attrs["auxiliary_signals"]]
            # TODO: add code to accumulate aux_signals
            self.accumulate_aux_signals = False if self.aux_signal_names == ["weight"] else True
            if "weight" in self.aux_signal_names:
                self.renormalisation = True
        else:
            self.aux_signal_names = None

        if "signal" in nxdata.attrs:
            signal_name = nxdata.attrs["signal"]
            self.signal_name = decode_to_string(signal_name)
        elif "data" in nxdata.keys():
            self.signal_name = "data"

        if hasattr(self, "signal_name") and "axes" in nxdata.attrs:
            self.axes_names = [decode_to_string(name) for name in nxdata.attrs["axes"]]
            # TODO: add logic to create axes if not in file (from integer sequence starting at 0)
            return
        raise KeyError

    def _accumulate_volumes(self) -> None:
        for i, data_file in enumerate(self.output_data_files):
            with h5py.File(data_file, "r") as f:
                volumes_array = np.array(f[self.nxentry_name][self.nxdata_name][self.signal_name])
                if self.renormalisation:
                    weights_array = np.array(f[self.nxentry_name][self.nxdata_name]["weight"])

            if self.renormalisation:
                volumes_array = np.multiply(volumes_array, weights_array)
                self.accumulator_weights[self.all_slices[i]] += weights_array
            self.accumulator_volume[self.all_slices[i]] += volumes_array

        if self.renormalisation:
            self.total_volume = self.accumulator_volume / self.accumulator_weights
            self.total_volume[np.isnan(self.total_volume)] = 0

    def _write_aggregation_file(self, aggregation_output_dir: Path, output_data_files: List[Path]) -> Path:
        aggregated_data_file = aggregation_output_dir / "aggregated_results.nxs"

        with h5py.File(aggregated_data_file, "w") as f:
            processed = f.create_group(self.nxentry_name)
            processed.attrs["NX_class"] = "NXentry"
            processed.attrs["default"] = self.nxdata_name

            process_name = "/".join((self.nxentry_name, "process"))
            process = f.create_group(process_name)
            process.attrs["NX_class"] = "NXprocess"
            f.create_dataset("/".join((process_name, "date")), data=str(datetime.now(timezone.utc)))
            f.create_dataset("/".join((process_name, "parameters")),
                             data=f"inputs: {output_data_files}, output: {aggregated_data_file}")
            f.create_dataset("/".join((process_name, "program")), data="ParProcCo")
            f.create_dataset("/".join((process_name, "version")), data=__version__)

            data_group_name = "/".join((self.nxentry_name, self.nxdata_name))
            data_group = f.create_group(data_group_name)
            data_group.attrs["NX_class"] = "NXdata"
            if self.renormalisation:
                data_group.attrs["auxiliary_signals"] = "weight"
            data_group.attrs["axes"] = self.axes_names
            data_group.attrs["signal"] = self.signal_name
            for i, axis in enumerate(self.axes_names):
                data_group.attrs[f"{axis}_indices"] = i
                f.create_dataset("/".join((data_group_name, f"{axis}")), data=self.accumulator_axis_ranges[i])
            f.create_dataset("/".join((data_group_name, self.signal_name)), data=self.total_volume)
            if self.renormalisation:
                f.create_dataset("/".join((data_group_name, "weight")), data=self.accumulator_weights)

            f.attrs["default"] = self.nxentry_name

            for i, filepath in enumerate(output_data_files):
                f[f"entry{i}"] = h5py.ExternalLink(str(filepath), self.nxentry_name)

            binoculars = f.create_group("binoculars")
            binoculars.attrs["type"] = "space"

            f.create_group("binoculars/axes")
            binocular_axes = [axis.split("-axis")[0].upper() for axis in self.axes_names]
            for i, axis in enumerate(binocular_axes):
                axis_min = self.axes_mins[i]
                axis_max = self.axes_maxs[i]
                scaling = (self.accumulator_axis_lengths[i] - 1) / (axis_max - axis_min)
                axis_dataset = [i, axis_min, axis_max, self.axes_spacing[i], axis_min * scaling, axis_max * scaling]
                f.create_dataset(f"binoculars/axes/{axis}", data=axis_dataset)

            binoculars["counts"] = f["/".join((data_group_name, self.signal_name))]
            if self.renormalisation:
                binoculars["contributions"] = f["/".join((data_group_name, "weight"))]

        return aggregated_data_file
