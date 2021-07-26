from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from simple_data_chunker import SimpleDataChunker


class TestDataChunker(unittest.TestCase):

    def setup_data_file(self, working_directory):
        # create test files
        file_name = "test_raw_data.txt"
        input_file_path = Path(working_directory) / file_name
        f = open(input_file_path, "w")
        f.write("3\n4\n11\n30\n")
        f.close()
        return input_file_path

    def test_chunk_data(self):
        base_dir = '/dls/tmp/vaq49247/tests/'
        with TemporaryDirectory(prefix='test_dir_', dir=base_dir) as working_directory:
            input_file_path = self.setup_data_file(working_directory)

            chunker = SimpleDataChunker(4)
            chunked_data_files = chunker.chunk(working_directory, input_file_path)

            self.assertEqual(len(chunked_data_files), 4)

            written_data = []
            for data_file in chunked_data_files:
                f = open(data_file, "r")
                lines = f.readlines()
                written_data.append(lines)
                f.close()

            self.assertEqual(written_data, [["3\n"], ["4\n"], ["11\n"], ["30\n"]])

    def test_aggregate_data(self):
        base_dir = '/dls/tmp/vaq49247/tests/'
        with TemporaryDirectory(prefix='test_dir_', dir=base_dir) as working_directory:
            cluster_output_dir = Path(working_directory) / "cluster_output"
            if not cluster_output_dir.exists():
                cluster_output_dir.mkdir(exist_ok=True, parents=True)

            input_file_path = self.setup_data_file(working_directory)

            chunker = SimpleDataChunker(4)
            chunked_data_files = chunker.chunk(working_directory, input_file_path)

            self.assertEqual(len(chunked_data_files), 4)

            written_data = []
            for data_file in chunked_data_files:
                f = open(data_file, "r")
                lines = f.readlines()
                written_data.append(lines)
                f.close()

            self.assertEqual(written_data, [["3\n"], ["4\n"], ["11\n"], ["30\n"]])

            agg_data_path = chunker.aggregate(cluster_output_dir, chunked_data_files)
            af = open(agg_data_path, "r")
            agg_data = af.readlines()
            af.close()

            self.assertEqual(agg_data, ["48"])


if __name__ == '__main__':
    unittest.main()
