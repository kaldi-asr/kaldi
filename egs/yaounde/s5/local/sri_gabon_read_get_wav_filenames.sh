#!/bin/bash
data=$1
dir=$2

mkdir -p $dir

find \
    $data \
    -type f \
    -name "*_read_*.wav" | \
    sort > \
	 \
	 $dir/sri_gabon_read_wav_filenames.txt
