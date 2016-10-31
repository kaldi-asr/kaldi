#!/bin/bash
data=$1
dir=$2

mkdir -p $dir

find $data -type f -name "*.wav" | sort -u > $dir/wav_filenames.txt
