#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from job_controller import JobController
from msm_data_slicer import MSMDataSlicer
from msmapper_aggregator import MSMAggregator


def create_parser():
    '''
     $ msm_cluster_submit rs_map --jobs --cores 6 --memory 4G -s 0.01 ... [input files]
     '''
    parser = argparse.ArgumentParser(description='Miller Space Mapper run script for use with ParProcCo')
    parser.add_argument('-o', '--output-dir', help='str: cluster output directory', required=True)
    parser.add_argument('-j', '--jobs', help='int: number of cluster jobs to split processing into', required=True)
    parser.add_argument('-f', '--file', help='str: input data file', required=True)
    return parser


def run_msm(args: argparse.Namespace, other_args: List) -> None:
    '''
    Run JobController
    '''
    jc = JobController(args.output_dir, "b24", "medium.q")
    args.jobs = int(args.jobs)
    args.file = Path(args.file)
    agg_data_path = jc.run(MSMDataSlicer(), MSMAggregator(), args.file, args.jobs, "msm_cluster_runner", other_args)
    print("complete")


if __name__ == '__main__':
    args, other_args = create_parser().parse_known_args()
    print(f"Parsing {args}")
    run_msm(args, other_args)
