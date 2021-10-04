#!/usr/bin/env python
# -*- coding: utf-8 -*-

import subprocess


def create_parser():
    import argparse
    parser = argparse.ArgumentParser(description='Miller Space Mapper run script for use with ParProcCo')
    parser.add_argument('--input_path', help='str: input data file', required=True)
    parser.add_argument('--output_path', help='str: output file path', required=True)
    parser.add_argument('-I', help='str: image slice parameter', required=True)
    parser.add_argument('-c', '--cpu', help='inr: cpus', default=None)
    parser.add_argument('-t', '--timeout', help='int: timeout (in minutes)', default=None)
    return parser


def run_rs_map(args) -> None:
    '''
    Run JobController
    '''
    args = ["-o", args.output_path, "-I", args.I, args.input_path, "rs_map"]
    proc = subprocess.Popen(args)
    proc.communicate()
    print("complete")


if __name__ == '__main__':
    args = create_parser().parse_args()
    print(f"Parsing {args}")
    run_rs_map(args)
