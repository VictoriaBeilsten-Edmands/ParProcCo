#!/bin/bash

if [[ ! $1 ]]; then
printf "Error: no input file provided: Exiting\n" >&2
	exit 1
fi
input_filepath="$1"
sleep 5
input_file_content=`cat "$input_filepath"`
let output_file_content=input_file_content*2

printf "$output_file_content"
