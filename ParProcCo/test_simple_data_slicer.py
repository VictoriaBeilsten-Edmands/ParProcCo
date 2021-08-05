import getpass
from pathlib import Path
from tempfile import TemporaryDirectory
import logging
import unittest

from simple_data_slicer import SimpleDataSlicer


def setup_data_file(working_directory):
    # create test files
    file_name = "test_raw_data.txt"
    input_file_path = Path(working_directory) / file_name
    with open(input_file_path, "w") as f:
        f.write("0\n1\n2\n3\n4\n5\n6\n7\n8\n9\n10\n")
    return input_file_path


class TestDataSlicer(unittest.TestCase):

    def setUp(self):
        current_user = getpass.getuser()
        tmp_dir = f"/dls/tmp/{current_user}/"
        self.base_dir = f"/dls/tmp/{current_user}/tests/"
        self.assertTrue(Path(tmp_dir).is_dir(), f"{tmp_dir} is not a directory")
        if not Path(self.base_dir).is_dir():
            logging.debug(f"Making directory {self.base_dir}")
            Path(self.base_dir).mkdir(exist_ok=True)

    def test_slice_params(self):
        with TemporaryDirectory(prefix='test_dir_', dir=self.base_dir) as working_directory:
            input_file_path = setup_data_file(working_directory)

            slicer = SimpleDataSlicer()
            slice_params = slicer.slice(input_file_path, 4, stop=8)

            self.assertEqual(len(slice_params), 4)

            self.assertEqual(slice_params, [["--start", "0", "--stop", "8", "--step", "4"],
                                            ["--start", "1", "--stop", "8", "--step", "4"],
                                            ["--start", "2", "--stop", "8", "--step", "4"],
                                            ["--start", "3", "--stop", "8", "--step", "4"]])

    def test_no_stop(self):
        with TemporaryDirectory(prefix='test_dir_', dir=self.base_dir) as working_directory:
            input_file_path = setup_data_file(working_directory)

            slicer = SimpleDataSlicer()
            slice_params = slicer.slice(input_file_path, 4)

            self.assertEqual(len(slice_params), 4)

            self.assertEqual(slice_params, [["--start", "0", "--stop", "11", "--step", "4"],
                                            ["--start", "1", "--stop", "11", "--step", "4"],
                                            ["--start", "2", "--stop", "11", "--step", "4"],
                                            ["--start", "3", "--stop", "11", "--step", "4"]])

    def test_too_many_slices(self):
        with TemporaryDirectory(prefix='test_dir_', dir=self.base_dir) as working_directory:
            input_file_path = setup_data_file(working_directory)

            slicer = SimpleDataSlicer()
            slice_params = slicer.slice(input_file_path, 20)

            self.assertEqual(len(slice_params), 11)

            self.assertEqual(slice_params, [["--start", "0", "--stop", "11", "--step", "11"],
                                            ["--start", "1", "--stop", "11", "--step", "11"],
                                            ["--start", "2", "--stop", "11", "--step", "11"],
                                            ["--start", "3", "--stop", "11", "--step", "11"],
                                            ["--start", "4", "--stop", "11", "--step", "11"],
                                            ["--start", "5", "--stop", "11", "--step", "11"],
                                            ["--start", "6", "--stop", "11", "--step", "11"],
                                            ["--start", "7", "--stop", "11", "--step", "11"],
                                            ["--start", "8", "--stop", "11", "--step", "11"],
                                            ["--start", "9", "--stop", "11", "--step", "11"],
                                            ["--start", "10", "--stop", "11", "--step", "11"]])

    def test_stop_too_big(self):
        with TemporaryDirectory(prefix='test_dir_', dir=self.base_dir) as working_directory:
            input_file_path = setup_data_file(working_directory)

            slicer = SimpleDataSlicer()
            slice_params = slicer.slice(input_file_path, 4, stop=20)

            self.assertEqual(len(slice_params), 4)

            self.assertEqual(slice_params, [["--start", "0", "--stop", "11", "--step", "4"],
                                            ["--start", "1", "--stop", "11", "--step", "4"],
                                            ["--start", "2", "--stop", "11", "--step", "4"],
                                            ["--start", "3", "--stop", "11", "--step", "4"]])


if __name__ == '__main__':
    unittest.main()
