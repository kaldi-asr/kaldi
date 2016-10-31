#!/bin/bash
data=$1
dir=$2

mkdir -p $dir

find $data -mindepth 1 -maxdepth 1 -type d -name "*sri_gabon*"    > $dir/speaker_directory_paths.txt 
