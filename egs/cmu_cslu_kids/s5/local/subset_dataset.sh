#!/usr/bin/env bash

# Copyright 2017  Luminar Technologies, Inc. (author: Daniel Galvez)
# Apache 2.0

# The following commands were used to generate the mini_librispeech dataset:
#
# Note that data generation is random. This could be fixed by
# providing a seed argument to the shuf program.

if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <src-dir> <dst-dir> <num-hours>"
  echo "e.g.: $0 /export/a05/dgalvez/LibriSpeech/train-clean-100 \\
                 /export/a05/dgalvez/LibriSpeech/train-clean-5 5"
  exit 1
fi

src_dir=$1
dest_dir=$2
dest_num_hours=$3

src=$(basename $src_dir)
dest=$(basename $dest_dir)
librispeech_dir=$(dirname $src_dir)

# TODO: Possibly improve this to ensure gender balance and speaker
# balance.
# TODO: Use actual time values instead of assuming that to make sure we get $dest_num_hours of data
src_num_hours=$(grep "$src" $librispeech_dir/CHAPTERS.TXT | awk -F'|' '{ print $3 }' | \
python -c '
from __future__ import print_function
from sys import stdin
minutes_str = stdin.read().split()
print(int(round(sum([float(minutes) for minutes in minutes_str]) / 60.0)))')
src_num_chapters=$(grep "$src" $librispeech_dir/CHAPTERS.TXT | \
                      awk -F'|' '{ print $1 }' | sort -u | wc -l)
mkdir -p data/subset_tmp
grep "$src" $librispeech_dir/CHAPTERS.TXT | \
  awk -F'|' '{ print $1 }' | \
  shuf -n $(((dest_num_hours * src_num_chapters) / src_num_hours)) > \
       data/subset_tmp/${dest}_chapter_id_list.txt

while read -r chapter_id || [[ -n "$chapter_id" ]]; do
  chapter_dir=$(find $src_dir/ -mindepth 2 -name "$chapter_id" -type d)
  speaker_id=$(basename $(dirname $chapter_dir))
  mkdir -p $dest_dir/$speaker_id/
  cp -r $chapter_dir $dest_dir/$speaker_id/
done  < data/subset_tmp/${dest}_chapter_id_list.txt
