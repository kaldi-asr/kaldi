#!/usr/bin/env bash

# Copyright  2020  Desh Raj (Johns Hopkins University)
# Apache 2.0

# This script trains an overlap detector. It is based on the Aspire
# speech activity detection system. Essentially this overlap
# detector is trained on whole recordings so it can be used to
# decode entire recordings without any SAD. We train with 3 targets:
# silence, single, and overlap. As such, at decode time, this
# can also be used as an SAD system.

affix=1a

train_stage=-10
stage=0
nj=50
test_nj=10

. ./cmd.sh

if [ -f ./path.sh ]; then . ./path.sh; fi

set -e -u -o pipefail
. utils/parse_options.sh 

if [ $# -ne 1 ]; then
  exit 1
fi

data_dir=$1
dir=exp/overlap_${affix}

whole_data_dir=${data_dir}_whole
whole_data_id=$(basename $data_dir)

mkdir -p $dir

if [ $stage -le 0 ]; then
  # First prepare the overlap RTTM from the training data
  # This will contain segments with the labels: single, overlap
  local/overlap/get_overlap_segments.py $data_dir/ref_rttm |\
    awk '{if ($8!="overlap"){$8="single"}{print}}' > $dir/overlap_rttm
fi

if [ $stage -le 1 ]; then
  # The training data may already be segmented, so we first prepare
  # a "whole" training data (not segmented) for training the overlap
  # detector.
  utils/data/convert_data_dir_to_whole.sh $data_dir $whole_data_dir
fi

###############################################################################
# Extract features for the whole data directory. We extract 40-dim MFCCs to 
# train the NN-based overlap detector.
###############################################################################
if [ $stage -le 2 ]; then
  mfcc_nj=$(wc -l < "$whole_data_dir/wav.scp")
  steps/make_mfcc.sh --nj $mfcc_nj --cmd "$train_cmd"  --write-utt2num-frames true \
    --mfcc-config conf/mfcc_hires.conf \
    ${whole_data_dir} exp/make_mfcc/${whole_data_id}
  steps/compute_cmvn_stats.sh ${whole_data_dir} exp/make_mfcc/${whole_data_id}
  utils/fix_data_dir.sh ${whole_data_dir}
fi

###############################################################################
# Prepare targets for training the overlap detector
###############################################################################
if [ $stage -le 3 ]; then
  frame_shift=$( cat ${whole_data_dir}/frame_shift ) 
  local/overlap/get_overlap_targets.py \
    --frame-shift $frame_shift \
    ${whole_data_dir}/utt2num_frames $dir/overlap_rttm - |\
    copy-feats ark,t:- ark,scp:$dir/targets.ark,$dir/targets.scp
  
fi

###############################################################################
# Train a neural network for SAD
###############################################################################
if [ $stage -le 4 ]; then
  # Train a STATS-pooling network for SAD
  local/overlap/train_tdnn_1a.sh \
    --stage 1 \
    --targets-dir $dir \
    --data-dir ${whole_data_dir} --affix "1a" || exit 1
fi

exit 0;
