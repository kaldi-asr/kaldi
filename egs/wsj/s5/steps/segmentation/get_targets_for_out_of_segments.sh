#!/usr/bin/env bash

# Copyright 2017  Vimal Manohar
# Apache 2.0

# This script prepares targets for whole recordings for training 
# speech activity detection system on the out-of-segment regions. 
# See the script steps/segmentation/lats_to_targets.sh for details about the 
# targets matrix.
# The out-of-segment regions are assigned the target values in the 
# file specified (in kaldi vector text format) by --default-targets option. 
# The in-segment regions are all assigned [ 0 0 0 ], 
# which means they don't contribute to the training. We will later be 
# combining these targets with other targets obtained from 
# supervision-constrained lattices and decoded lattices using the 
# script steps/segmentation/merge_targets.sh.
# By default, the 'default_targets' would be [ 1 0 0 ], which means all
# the out-of-segment regions are assumed as silence. But depending, on
# the application and data, this could be [ 0 0 0 ] or [ 0 0 1 ] or
# something with fractional weights.

nj=4
cmd=run.pl
default_targets=   # vector of default targets in text format
frame_subsampling_factor=1

set -o pipefail -u

[ -f ./path.sh ] && . ./path.sh
. utils/parse_options.sh

if [ $# -ne 3 ]; then
  cat <<EOF
  This script prepares targets for whole recordings for training 
  speech activity detection system on the out-of-segment regions. 
  See the top of the script for details.
  Usage: steps/segmentation/get_targets_for_out_of_segments.sh <data-dir> <whole-data-dir> <targets-dir>
   e.g.: steps/segmentation/get_targets_for_out_of_segments.sh \
    data/train_split10s data/train_whole \
    exp/segmentation1a/out_of_train_split10s_train_whole_default_targets
EOF
  exit 1
fi

data=$1
whole_data=$2
dir=$3

for f in $data/segments $whole_data/wav.scp; do
  if [ ! -f $f ]; then 
    echo "$0: Could not find file $f" 
    exit 1
  fi
done

frame_shift=$(utils/data/get_frame_shift.sh $data) || exit 1

mkdir -p $dir/split${nj}reco
split_scps=
for n in $(seq $nj); do
  split_scps="$split_scps $dir/split${nj}reco/wav.$n.scp"
done
utils/split_scp.pl $whole_data/wav.scp $split_scps

utils/data/get_reco2utt_for_data.sh $data > $dir/reco2utt

mkdir -p $dir/split${nj}reco
utils/filter_scps.pl JOB=1:$nj $dir/split${nj}reco/wav.JOB.scp $dir/reco2utt \
  $dir/split${nj}reco/reco2utt.JOB || exit 1
utils/filter_scps.pl -f 2 JOB=1:$nj $dir/split${nj}reco/wav.JOB.scp $data/segments \
    $dir/split${nj}reco/segments.JOB || exit 1

# make $dir an absolute pathname.
dir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' $dir ${PWD}`

utils/data/get_utt2num_frames.sh $whole_data
cp $whole_data/utt2num_frames $dir/reco2num_frames

$cmd JOB=1:$nj $dir/log/get_default_targets.JOB.log \
  steps/segmentation/internal/get_default_targets_for_out_of_segments.py \
    --reco2num-frames=$dir/reco2num_frames \
    --default-targets="$default_targets" \
    $dir/split${nj}reco/reco2utt.JOB $dir/split${nj}reco/segments.JOB - \| \
  subsample-feats --n=$frame_subsampling_factor ark,t:- ark:- \| \
  copy-feats ark:- ark,scp:$dir/targets.JOB.ark,$dir/targets.JOB.scp || exit 1

if [ $frame_subsampling_factor -ne 1 ]; then
  echo $frame_subsampling_factor > $dir/frame_subsampling_factor
fi

for n in $(seq $nj); do
  cat $dir/targets.$n.scp
done | sort -k1,1 > $dir/targets.scp

steps/segmentation/validate_targets_dir.sh $dir $whole_data || exit 1

echo "$0: Got default targets for out-of-segments regions in $whole_data corresponding to segments in $data"

exit 0
