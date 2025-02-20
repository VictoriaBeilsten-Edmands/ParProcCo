from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple

from ParProcCo.scheduler_mode_interface import SchedulerModeInterface
from ParProcCo.utils import check_jobscript_is_readable, check_location, get_absolute_path

class NXdataAggregationMode(SchedulerModeInterface):
    def __init__(self):
        current_script_dir = Path(os.path.realpath(__file__)).parent.parent / "scripts"
        self.program_path = current_script_dir / "nxdata_aggregate"
        self.cores = 1
        self.allowed_modules = ('python',)

    def set_parameters(self, sliced_results: List[Path]) -> None:
        """Overrides SchedulerModeInterface.set_parameters"""
        self.sliced_results = [str(res) for res in sliced_results]
        self.number_jobs: int = 1

    def generate_output_paths(self, output_dir: Optional[Path], error_dir: Path, i: int) -> Tuple[str, str, str]:
        """Overrides SchedulerModeInterface.generate_output_paths"""
        output_file = "aggregated_results.nxs"
        output_fp = str(output_dir / output_file) if output_dir else output_file
        stdout_fp = str(error_dir / "out_aggregated")
        stderr_fp = str(error_dir / "err_aggregated")
        return output_fp, stdout_fp, stderr_fp

    def generate_args(self, i: int, _memory: str, _cores: int, jobscript_args: List[str],
                      output_fp: str) -> Tuple[str, ...]:
        """Overrides SchedulerModeInterface.generate_args"""
        assert(i == 0)
        jobscript = str(check_jobscript_is_readable(check_location(get_absolute_path(jobscript_args[0]))))
        args = tuple([jobscript, "--output", output_fp] + self.sliced_results)
        return args
