
class SimpleDataSlicer:

    def __init__(self):
        pass

    def slice(self, input_data_file, number_jobs, stop=None):
        if type(number_jobs) is not int:
            raise TypeError(f"number_jobs is {type(number_jobs)}, should be int\n")

        file_length = sum(1 for line in open(input_data_file))

        if stop is None:
            stop = file_length
        elif type(stop) is not int:
            raise TypeError(f"stop is {type(stop)}, should be int\n")
        else:
            stop = min(stop, file_length)

        number_jobs = min(stop, number_jobs)

        slice_params = [["--start", str(i), "--stop", str(stop), "--step", str(number_jobs)]
                        for i in range(number_jobs)]
        return slice_params
