#!/bin/bash
# Copyright 2015  Brno University of Technology (Author: Karel Vesely)
# Licensed under the Apache License, Version 2.0 (the "License")

# Extract lattice-depth for each frame.

# Begin configuration
cmd=run.pl
# End configuration

echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh # source the path.
. parse_options.sh || exit 1;

if [ $# != 2 ]; then
   echo "usage: $0 [opts] <dir-with-lats> <out-dir>"
   echo "main options (for others, see top of script file)"
   echo "  --config <config-file>          # config containing options"
   echo "  --cmd"
   exit 1;
fi

set -euo pipefail

latdir=$1
dir=$2

[ ! -f $latdir/lat.1.gz ] && echo "Missing $latdir/lat.1.gz" && exit 1
nj=$(cat $latdir/num_jobs)

# Get the pdf-posterior vectors,
$cmd JOB=1:$nj $dir/log/lattice_depth_per_frame.JOB.log \
  lattice-depth-per-frame "ark:gunzip -c $latdir/lat.JOB.gz |" ark,t:$dir/lattice_frame_depth.JOB.ark
# Merge,
for ((n=1; n<=nj; n++)); do cat $dir/lattice_frame_depth.${n}.ark; done >$dir/lattice_frame_depth.ark
rm $dir/lattice_frame_depth.*.ark

# Done!
