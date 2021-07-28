import getpass
from pathlib import Path
from tempfile import TemporaryDirectory
import logging
import unittest

from simple_data_chunker import SimpleDataChunker


def setup_data_file(working_directory):
    # create test files
    file_name = "test_raw_data.txt"
    input_file_path = Path(working_directory) / file_name
    with open(input_file_path, "w") as f:
        f.write("3\n4\n11\n30\n")
    return input_file_path


class TestDataChunker(unittest.TestCase):

    def setUp(self):
        current_user = getpass.getuser()
        tmp_dir = f"/dls/tmp/{current_user}/"
        self.base_dir = f"/dls/tmp/{current_user}/tests/"
        self.assertTrue(Path(tmp_dir).is_dir(), f"{tmp_dir} is not a directory")
        if not Path(self.base_dir).is_dir():
            logging.debug(f"Making directory {self.base_dir}")
            Path(self.base_dir).mkdir(exist_ok=True)

    def test_chunk_data(self):
        with TemporaryDirectory(prefix='test_dir_', dir=self.base_dir) as working_directory:
            input_file_path = setup_data_file(working_directory)

            chunker = SimpleDataChunker(4)
            chunked_data_files = chunker.chunk(working_directory, input_file_path)

            self.assertEqual(len(chunked_data_files), 4)

            written_data = []
            for data_file in chunked_data_files:
                with open(data_file, "r") as f:
                    lines = f.readlines()
                    written_data.append(lines)

            self.assertEqual(written_data, [["3\n"], ["4\n"], ["11\n"], ["30\n"]])

    def test_too_many_chunks(self):
        with TemporaryDirectory(prefix='test_dir_', dir=self.base_dir) as working_directory:
            input_file_path = setup_data_file(working_directory)

            chunker = SimpleDataChunker(7)
            chunked_data_files = chunker.chunk(working_directory, input_file_path)

            self.assertEqual(len(chunked_data_files), 4)

            written_data = []
            for data_file in chunked_data_files:
                with open(data_file, "r") as f:
                    lines = f.readlines()
                    written_data.append(lines)

            self.assertEqual(written_data, [["3\n"], ["4\n"], ["11\n"], ["30\n"]])

    def test_aggregate_data(self):
        with TemporaryDirectory(prefix='test_dir_', dir=self.base_dir) as working_directory:
            cluster_output_dir = Path(working_directory) / "cluster_output"
            if not cluster_output_dir.exists():
                cluster_output_dir.mkdir(exist_ok=True, parents=True)

            input_file_path = setup_data_file(working_directory)

            chunker = SimpleDataChunker(4)
            chunked_data_files = chunker.chunk(working_directory, input_file_path)

            self.assertEqual(len(chunked_data_files), 4)

            written_data = []
            for data_file in chunked_data_files:
                with open(data_file, "r") as f:
                    lines = f.readlines()
                    written_data.append(lines)

            self.assertEqual(written_data, [["3\n"], ["4\n"], ["11\n"], ["30\n"]])

            agg_data_path = chunker.aggregate(cluster_output_dir, chunked_data_files)
            with open(agg_data_path, "r") as af:
                agg_data = af.readlines()

            self.assertEqual(agg_data, ["48"])


if __name__ == '__main__':
    unittest.main()
