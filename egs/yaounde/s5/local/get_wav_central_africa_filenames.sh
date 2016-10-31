#!/bin/bash
data=$1
tmp_dir=data/local/tmp
mkdir -p ${tmp_dir}
find $data \
     -mindepth 1 \
     -maxdepth 1 \
     -type d \
     > \
     ${tmp_dir}/speaker_directory_paths.txt 

find $data \
     -type f \
     -name "*.wav"  \
    | \
    sort -u \
	 > \
	 ${tmp_dir}/wav_filenames.txt
