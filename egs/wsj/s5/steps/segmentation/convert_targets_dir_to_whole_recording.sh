#!/usr/bin/env bash

# Copyright 2017  Vimal Manohar
# Apache 2.0

# This script converts targets corresponding to 'data' at segments level 
# in 'targets_dir' to whole-recording level corresponding to the 
# whole-recording data directory 'whole_data'.

# The targets for the whole-recording are created by simply copying the targets 
# for the in-segment region, while setting the out-of-segment region targets
# to the target values contained in the file specified 
# (in kaldi vector text format) by --default-targets option.
# By default, the 'default_targets' would be [ 0 0 0 ].
# Note that the script steps/segmentation/get_targets_for_out_of_segments.sh 
# can be used to get targets only for the out-of-segment regions. It is 
# better to use that when you need specific target values like all silence 
# ([ 1 0 0 ]) or all garbage ([ 0 0 1 ]) for the out-of-segment regions. 
# That way you can control how the out-of-segment target values are 
# combined using the weights in steps/segmentation/merge_targets_dirs.sh

nj=4
cmd=run.pl
default_targets=   # vector of default targets in text format

set -o pipefail -u

[ -f ./path.sh ] && . ./path.sh
. utils/parse_options.sh

if [ $# -ne 4 ]; then
  cat <<EOF
  This script converts targets corresponding to 'data' at segments level 
  in 'targets_dir' to whole-recording level corresponding to the 
  whole-recording data directory 'whole_data'.
  See top of the script for more details.

  Usage: steps/segmentation/convert_targets_to_whole_recording.sh <data-dir> <whole-data-dir> <targets-dir> <whole-targets-dir>
   e.g.: steps/segmentation/convert_targets_to_whole_recording.sh \
    data/train_split10s data/train_whole \
    exp/segmentation1a/tri3b_train_split10s_targets \
    exp/segmentation1a/tri3b_train_whole_targets
EOF
  exit 1
fi

data=$1
whole_data=$2
targets_dir=$3
dir=$4

if [ ! -f $data/segments ]; then
  awk '{print $1}' $whole_data/wav.scp > $dir/recos
  utils/filter_scp.pl $data/utt2spk $dir/recos > $dir/recos.data

  nr=$(cat $dir/reco | wc -l)
  nu=$(cat $dir/recos.data | wc -l) 

  if [ $nu -lt $[$nr - ($nr/20)] ]; then
    echo "Found less that 95% the recordings of $whole_data in $data."
    exit 1;
  fi

  cp $targets_dir/targets.scp $dir
  cp $targets_dir/frame_subsampling_factor $dir || true

  exit 0
fi

for f in $data/segments $targets_dir/targets.scp \
  $whole_data/wav.scp; do
  if [ ! -f $f ]; then 
    echo "$0: Could not find file $f" 
    exit 1
  fi
done

frame_shift=$(utils/data/get_frame_shift.sh $data) || exit 1
frame_subsampling_factor=1
if [ -f $targets_dir/frame_subsampling_factor ]; then
  frame_subsampling_factor=$(cat $targets_dir/frames_subsampling_factor) || exit 1
fi
frame_shift=`perl -e "print ($frame_shift * $frame_subsampling_factor);"`

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
utils/filter_scps.pl JOB=1:$nj $dir/split${nj}reco/segments.JOB $targets_dir/targets.scp \
    $dir/split${nj}reco/targets.JOB.scp || exit 1

# make $dir an absolute pathname.
dir=`perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' $dir ${PWD}`

utils/data/get_utt2num_frames.sh --cmd "$cmd" --nj $nj $whole_data
cp $whole_data/utt2num_frames $dir/reco2num_frames

$cmd JOB=1:$nj $dir/log/merge_targets_to_reco.JOB.log \
  steps/segmentation/internal/merge_segment_targets_to_recording.py \
    --reco2num-frames=$dir/reco2num_frames --frame-shift=$frame_shift \
    --default-targets="$default_targets" \
    $dir/split${nj}reco/reco2utt.JOB $dir/split${nj}reco/segments.JOB \
    $dir/split${nj}reco/targets.JOB.scp - \| \
  copy-feats ark,t:- ark,scp:$dir/targets.JOB.ark,$dir/targets.JOB.scp || exit 1

for n in $(seq $nj); do
  cat $dir/targets.$n.scp
done | sort -k1,1 > $dir/targets.scp

steps/segmentation/validate_targets_dir.sh $dir $whole_data || exit 1

echo "$0: Converted targets to whole recordings in $dir"
exit 0
