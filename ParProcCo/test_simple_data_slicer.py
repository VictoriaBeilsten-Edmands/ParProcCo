from __future__ import annotations

import getpass
import logging
import unittest
from pathlib import Path
from parameterized import parameterized
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

    @parameterized.expand([
        ("all_ok", [4], {"stop": 8}, None, 4, [slice(0, 8, 4), slice(1, 8, 4), slice(2, 8, 4), slice(3, 8, 4)]),
        ("no_stop", [4], {}, None, 4, [slice(0, 11, 4), slice(1, 11, 4), slice(2, 11, 4), slice(3, 11, 4)]),
        ("stop_not_int", [4], {"stop": "8"}, "stop is <class 'str'>, should be int", None, None),
        ("number_jobs_not_int", ["4"], {}, "number_jobs is <class 'str'>, should be int", None, None),
        ("too_many_slices", [20], {}, None, 11,
         [slice(0, 11, 11), slice(1, 11, 11), slice(2, 11, 11), slice(3, 11, 11), slice(4, 11, 11), slice(5, 11, 11),
          slice(6, 11, 11), slice(7, 11, 11), slice(8, 11, 11), slice(9, 11, 11), slice(10, 11, 11)]),
        ("stop_too_big", [4], {"stop": 20}, None, 4,
         [slice(0, 11, 4), slice(1, 11, 4), slice(2, 11, 4), slice(3, 11, 4)])
    ])
    def test_slices(self, name, args, kwargs, error_msg, expected_length, expected_slices) -> None:
        with TemporaryDirectory(prefix='test_dir_', dir=self.base_dir) as working_directory:
            input_file_path = setup_data_file(working_directory)

            slicer = SimpleDataSlicer()
            args.insert(0, input_file_path)

            if error_msg:
                with self.assertRaises(TypeError) as context:
                    slicer.slice(*args, **kwargs)
                self.assertTrue(error_msg in str(context.exception))
                return

            slices = slicer.slice(*args, **kwargs)

            self.assertEqual(len(slices), expected_length)
            self.assertEqual(slices, expected_slices)


if __name__ == '__main__':
    unittest.main()
