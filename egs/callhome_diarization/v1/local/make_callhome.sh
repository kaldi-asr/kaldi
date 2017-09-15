#!/bin/bash
# Copyright 2017   David Snyder
# Apache 2.0.
#
# TODO, this is a temporary data prep script for Callhome.

set -e

src_dir=$1
data_dir=$2

mkdir -p $data_dir/callhome
mkdir -p $data_dir/callhome1
mkdir -p $data_dir/callhome2

for name in callhome callhome1 callhome2; do
  cp $src_dir/wav.scp $data_dir/$name
  cp $src_dir/segments $data_dir/$name
  cp $src_dir/utt2spk $data_dir/$name
  cp $src_dir/spk2utt $data_dir/$name
  cp $src_dir/reco2num $data_dir/$name
done

utils/validate_data_dir.sh --no-text --no-feats $data_dir/callhome
utils/fix_data_dir.sh $data_dir/callhome

shuf $data_dir/callhome/wav.scp | head -n 250 | utils/filter_scp.pl - $data_dir/callhome/wav.scp > $data_dir/callhome1/wav.scp
utils/fix_data_dir.sh $data_dir/callhome1
utils/filter_scp.pl --exclude $data_dir/callhome1/wav.scp $data_dir/callhome/wav.scp > $data_dir/callhome2/wav.scp
utils/fix_data_dir.sh $data_dir/callhome2
utils/filter_scp.pl $data_dir/callhome1/wav.scp $data_dir/callhome/reco2num > $data_dir/callhome1/reco2num
utils/filter_scp.pl $data_dir/callhome2/wav.scp $data_dir/callhome/reco2num > $data_dir/callhome2/reco2num

cp $src_dir/fullref.rttm local/fullref.rttm
