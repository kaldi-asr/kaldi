#!/usr/bin/env bash

# Copyright 2017 Nagendra Kumar Goel
#           2018 Vimal Manohar   
# Apache 2.0.

# This script combines targets directory into a new targets directory 
# containing targets from all the input targets directories.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# -lt 3 ]; then
  echo "Usage: $0 [options] <data> <dest-targets-dir> <src-targets-dir1> <src-targets-dir2> ..."
  echo "e.g.: $0 data/train exp/targets_combined exp/targets_1 exp/targets_2"
  exit 1;
fi

export LC_ALL=C

data=$1;
shift;
dest=$1;
shift;
first_src=$1;

mkdir -p $dest;
rm -f $dest/{targets.*.ark,frame_subsampling_factor} 2>/dev/null

frame_subsampling_factor=1
if [ -f $first_src/frame_subsampling_factor ]; then
  cp $first_src/frame_subsampling_factor $dest
  frame_subsampling_factor=$(cat $dest/frame_subsampling_factor)
fi

for d in $*; do
  this_frame_subsampling_factor=1
  if [ -f $d/frame_subsampling_factor ]; then
    this_frame_subsampling_factor=$(cat $d/frame_subsampling_factor)
  fi

  if [ $this_frame_subsampling_factor != $frame_subsampling_factor ]; then
    echo "$0: Cannot combine targets directories with different frame-subsampling-factors" 1>&2
    exit 1
  fi

  cat $d/targets.scp
done | sort -k1,1 > $dest/targets.scp || exit 1

steps/segmentation/validate_targets_dir.sh $dest $data || exit 1

echo "Combined targets and stored in $dest"
exit 0
