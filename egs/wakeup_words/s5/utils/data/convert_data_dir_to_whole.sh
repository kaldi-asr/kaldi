#! /bin/bash

# Copyright 2016-2018  Vimal Manohar
# Apache 2.0

# This scripts converts a data directory into a "whole" data directory
# by removing the segments and using the recordings themselves as 
# utterances

set -o pipefail

. ./path.sh

. utils/parse_options.sh

if [ $# -ne 2 ]; then
  echo "Usage: convert_data_dir_to_whole.sh <in-data> <out-data>"
  echo " e.g.: convert_data_dir_to_whole.sh data/dev data/dev_whole"
  exit 1
fi

data=$1
dir=$2

if [ ! -f $data/segments ]; then
  echo "$0: Data directory already does not contain segments. So just copying it."
  utils/copy_data_dir.sh $data $dir
  exit 0
fi

mkdir -p $dir
cp $data/wav.scp $dir
if [ -f $data/reco2file_and_channel ]; then 
  cp $data/reco2file_and_channel $dir; 
fi

mkdir -p $dir/.backup
if [ -f $dir/feats.scp ]; then
  mv $dir/feats.scp $dir/.backup
fi
if [ -f $dir/cmvn.scp ]; then
  mv $dir/cmvn.scp $dir/.backup
fi
if [ -f $dir/utt2spk ]; then
  mv $dir/utt2spk $dir/.backup
fi

[ -f $data/stm ] && cp $data/stm $dir
[ -f $data/glm ] && cp $data/glm $dir

utils/data/internal/combine_segments_to_recording.py \
  --write-reco2utt=$dir/reco2sorted_utts $data/segments $dir/utt2spk || exit 1

if [ -f $data/text ]; then
  utils/apply_map.pl -f 2- $data/text < $dir/reco2sorted_utts > $dir/text || exit 1
fi

rm $dir/reco2sorted_utts

utils/fix_data_dir.sh $dir || exit 1

exit 0
