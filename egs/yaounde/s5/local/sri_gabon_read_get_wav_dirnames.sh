#!/bin/bash
data=$1
dir=$2

mkdir -p $dir

find \
    $data \
    -mindepth 1 \
    -maxdepth 1 \
    -type d      | \
    grep read > \
    $dir/sri_gabon_read_speaker_directory_paths.txt 
