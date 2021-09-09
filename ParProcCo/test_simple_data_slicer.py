from __future__ import annotations

import getpass
import logging
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from simple_data_slicer import SimpleDataSlicer


def setup_data_file(working_directory: str) -> Path:
    # create test files
    file_name = "test_raw_data.txt"
    input_file_path = Path(working_directory) / file_name
    with open(input_file_path, "w") as f:
        f.write("0\n1\n2\n3\n4\n5\n6\n7\n8\n9\n10\n")
    return input_file_path


class TestDataSlicer(unittest.TestCase):

    def setUp(self) -> None:
        current_user = getpass.getuser()
        tmp_dir = f"/dls/tmp/{current_user}/"
        self.base_dir = f"/dls/tmp/{current_user}/tests/"
        self.assertTrue(Path(tmp_dir).is_dir(), f"{tmp_dir} is not a directory")
        if not Path(self.base_dir).is_dir():
            logging.debug(f"Making directory {self.base_dir}")
            Path(self.base_dir).mkdir(exist_ok=True)

    def test_slices(self) -> None:
        with TemporaryDirectory(prefix='test_dir_', dir=self.base_dir) as working_directory:
            input_file_path = setup_data_file(working_directory)

            slicer = SimpleDataSlicer()
            slices = slicer.slice(input_file_path, 4, stop=8)

            self.assertEqual(len(slices), 4)

            self.assertEqual(slices, [slice(0, 8, 4), slice(1, 8, 4), slice(2, 8, 4), slice(3, 8, 4)])

    def test_no_stop(self) -> None:
        with TemporaryDirectory(prefix='test_dir_', dir=self.base_dir) as working_directory:
            input_file_path = setup_data_file(working_directory)

            slicer = SimpleDataSlicer()
            slices = slicer.slice(input_file_path, 4)

            self.assertEqual(len(slices), 4)

            self.assertEqual(slices, [slice(0, 11, 4), slice(1, 11, 4), slice(2, 11, 4), slice(3, 11, 4)])

    def test_stop_not_int(self) -> None:
        with TemporaryDirectory(prefix='test_dir_', dir=self.base_dir) as working_directory:
            input_file_path = setup_data_file(working_directory)

            slicer = SimpleDataSlicer()

            with self.assertRaises(TypeError) as context:
                slicer.slice(input_file_path, 4, stop="8")
            self.assertTrue("stop is <class 'str'>, should be int" in str(context.exception))

    def test_number_jobs_not_int(self) -> None:
        with TemporaryDirectory(prefix='test_dir_', dir=self.base_dir) as working_directory:
            input_file_path = setup_data_file(working_directory)

            slicer = SimpleDataSlicer()

            with self.assertRaises(TypeError) as context:
                slicer.slice(input_file_path, "4")
            self.assertTrue("number_jobs is <class 'str'>, should be int" in str(context.exception))

    def test_too_many_slices(self) -> None:
        with TemporaryDirectory(prefix='test_dir_', dir=self.base_dir) as working_directory:
            input_file_path = setup_data_file(working_directory)

            slicer = SimpleDataSlicer()
            slices = slicer.slice(input_file_path, 20)

            self.assertEqual(len(slices), 11)

            self.assertEqual(slices, [slice(0, 11, 11), slice(1, 11, 11), slice(2, 11, 11), slice(3, 11, 11),
                                      slice(4, 11, 11), slice(5, 11, 11), slice(6, 11, 11), slice(7, 11, 11),
                                      slice(8, 11, 11), slice(9, 11, 11), slice(10, 11, 11)])

    def test_stop_too_big(self) -> None:
        with TemporaryDirectory(prefix='test_dir_', dir=self.base_dir) as working_directory:
            input_file_path = setup_data_file(working_directory)

            slicer = SimpleDataSlicer()
            slices = slicer.slice(input_file_path, 4, stop=20)

            self.assertEqual(len(slices), 4)

            self.assertEqual(slices, [slice(0, 11, 4), slice(1, 11, 4), slice(2, 11, 4), slice(3, 11, 4)])


if __name__ == '__main__':
    unittest.main()
