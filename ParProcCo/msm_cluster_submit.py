#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
from typing import List

from job_controller import JobController
from msm_data_slicer import MSMDataSlicer
from msmapper_aggregator import MSMAggregator
from argparse import ArgumentParser, Namespace


def create_parser():
    import argparse
    parser = argparse.ArgumentParser(description='Miller Space Mapper run script for use with ParProcCo')
    parser.add_argument('-o', '--output-dir', help='str: cluster output directory', required=True)
    parser.add_argument('-p', '--project', help='str: project', required=True)
    parser.add_argument('-q', '--queue', help='str: queue', required=True)
    parser.add_argument('-n', '--number-jobs', help='int: number of cluster jobs to split processing into', required=True)
    parser.add_argument('-f', '--file', help='str: input data file', required=True)
    parser.add_argument('-c', '--cpu', help='inr: cpus', default=None)
    parser.add_argument('-t', '--timeout', help='int: timeout (in minutes)', default=None)
    return parser


def run_msm(args: Namespace) -> None:
    '''
    Run JobController
    '''
    jc = JobController(args.output_dir, args.project, args.queue)
    args.number_jobs = int(args.number_jobs)
    args.file = Path(args.file)
    agg_data_path = jc.run(MSMDataSlicer(), MSMAggregator(), args.file, args.number_jobs, "rs_map_prefix.py")
    print("complete")


if __name__ == '__main__':
    args = create_parser().parse_args()
    print(f"Parsing {args}")
    run_msm(args)
