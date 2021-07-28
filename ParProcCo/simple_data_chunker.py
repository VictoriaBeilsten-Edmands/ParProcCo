from pathlib import Path


class SimpleDataChunker:

    def __init__(self, total_chunks):
        if type(total_chunks) is int:
            self.total_chunks = total_chunks
        else:
            raise TypeError(f"total_chunks is {type(total_chunks)}, should be int\n")

    def chunk(self, working_directory, input_data_file):
        chunked_data_files = []
        with open(input_data_file) as f:
            lines = f.readlines()
            total_lines = len(lines)
            self.total_chunks = min(self.total_chunks, total_lines)
            lines_per_chunk = total_lines // self.total_chunks
            remainder = total_lines % self.total_chunks
            line_chunk_totals = [(lines_per_chunk + 1) if j <
                                 remainder else lines_per_chunk for j in range(self.total_chunks)]

            current_chunk = 0
            counter = 0
            for line in lines:
                if counter == line_chunk_totals[current_chunk]:
                    current_chunk += 1
                    counter = 0

                chunked_file_name = f"chunked_data_{current_chunk}.txt"
                chunked_file_path = Path(working_directory) / chunked_file_name

                with open(chunked_file_path, "w") as sf:
                    sf.write(line)

                chunked_data_files.append(chunked_file_path)
                counter += 1

        return chunked_data_files

    def aggregate(self, cluster_output_dir, output_data_files):
        aggregated_data_file = Path(cluster_output_dir) / "aggregated_results.txt"

        total = 0
        for output_file in output_data_files:
            with open(output_file) as f:
                for line in f.readlines():
                    total += int(line.strip())

        with open(aggregated_data_file, "w") as af:
            af.write(str(total))

        return aggregated_data_file
