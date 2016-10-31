#!/bin/bash
data=$1
dir=$2

mkdir -p $dir

find \
    $data \
    -type f \
    -name "*_conv_*.wav" | \
    sort > \
	 \
	 $dir/sri_gabon_conv_wav_filenames.txt
