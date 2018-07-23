#!/bin/bash

# Copyright 2017  Vimal Manohar
# Apache 2.0

# This script resamples the targets matrix by the specified <subsampling-factor>.
# If <subsampling-factor> is negative, then the targets will be upsampled 
# by -<subsampling-factor>.
# This script is a wrapper to steps/segmentation/interal/resample_targets.py,
# which works very similar to the binary subsample-feats. See that script
# for details about how the resampling is done.

# See the script steps/segmentation/lats_to_targets.sh for details about 
# the format of the targets.

nj=4
cmd=run.pl

set -o pipefail -u

[ -f ./path.sh ] && . ./path.sh
. utils/parse_options.sh

if [ $# -ne 4 ]; then
  cat <<EOF
  This script resamples the targets matrix by the specified subsampling factor.
  If <subsampling-factor> is negative, then the targets will be upsampled 
  by -<subsampling-factor>.
  See top of the script for more details.

  Usage: steps/segmentation/resample_targets.sh <subsampling-factor> <data-dir> <targets-dir> <resampled-targets-dir>
   e.g.: steps/segmentation/resample_targets.sh 3 \
    data/train_whole \
    exp/segmentation1a/tri3b_train_whole_targets \
    exp/segmentation1a/tri3b_train_whole_targets_sub3
EOF
  exit 1
fi

subsampling_factor=$1
data=$2
targets_dir=$3
dir=$4

frame_subsampling_factor=1
if [ -f $targets_dir/frame_subsampling_factor ]; then
  frame_subsampling_factor=$(cat $targets_dir/frames_subsampling_factor)
fi

for f in $targets_dir/targets.scp $data/feats.scp; do 
  if [ ! -f $f ]; then 
    echo "$0: Could not find file $f" 
    exit 1
  fi
done

steps/segmentation/validate_targets_dir.sh $targets_dir $data || exit 1

mkdir -p $dir

mkdir -p $targets_dir/split$nj
split_scps=
for n in $(seq $nj); do
  split_scps="$split_scps $targets_dir/split${nj}/targets.$n.scp"
done
utils/split_scp.pl $targets_dir/targets.scp $split_scps

# make $dir an absolute pathname.
dir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' $dir ${PWD}`

if [ $subsampling_factor -eq 1 ]; then
  cp $targets_dir/targets.scp $dir
  cp $targets_dir/frame_subsampling_factor $dir || true
elif [ $subsampling_factor -gt 1 ]; then
  $cmd JOB=1:$nj $dir/log/resample_targets.JOB.log \
    copy-feats scp:$targets_dir/split${nj}/targets.JOB.scp ark,t:- \| \
    steps/segmentation/internal/resample_targets.py \
      --subsampling-factor=$subsampling_factor \
      - - \| \
    copy-feats ark,t:- ark,scp:$dir/targets.JOB.ark,$dir/targets.JOB.scp || exit 1

  perl -e "print $frame_subsampling_factor * $subsampling_factor" > \
    $dir/frame_subsampling_factor || exit 1
else
  $cmd JOB=1:$nj $dir/log/resample_targets.JOB.log \
    subsample-feats --n=$subsampling_factor \
      scp:$targets_dir/split${nj}/targets.JOB.scp \
      ark,scp:$dir/targets.JOB.ark,$dir/targets.JOB.scp || exit 1

  perl -e "print $frame_subsampling_factor * (-$subsampling_factor)" > \
    $dir/frame_subsampling_factor || exit 1
fi 
 
for n in $(seq $nj); do
  cat $dir/targets.$n.scp
done > $dir/targets.scp

steps/segmentation/validate_targets_dir.sh $targets_dir $data

echo "$0: Resampled targets in $dir"
exit 0
