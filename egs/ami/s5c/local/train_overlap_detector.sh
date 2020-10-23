#!/usr/bin/env bash

# Copyright  2020  Desh Raj (Johns Hopkins University)
# Apache 2.0

# This script trains an overlap detector. It is based on the Aspire
# speech activity detection system. We train with 3 targets:
# silence, single, and overlap. As such, at decode time, this
# can also be used as an SAD system. 

# We can use the annotated speech time marks or forced alignments 
# for training. Here we provide code for both. To use forced alignments
# we need a pretrained acoustic model in order to obtain the 
# alignments.

affix=1a

train_stage=-10
stage=0
nj=50
test_nj=10

test_sets="dev test"

target_type=annotation  # set this to "annotation" or "forced"

# If target_type is forced, the following must contain path to a tri3 model
src_dir=exp/tri3_cleaned
ali_dir=${src_dir}_ali

. ./cmd.sh

if [ -f ./path.sh ]; then . ./path.sh; fi

set -e -u -o pipefail
. utils/parse_options.sh 

if [ $# != 1 ]; then
  echo "Usage: $0 <ami-corpus-dir>"
  echo "e.g.: $0 /export/data/ami"
  echo "Options: "
  echo "  --nj <nj>                                        # number of parallel jobs."
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  exit 1;
fi

AMI_DIR=$1

train_set=train_ovl
dir=exp/overlap_${affix}

train_data_dir=data/${train_set}
whole_data_dir=data/${train_set}_whole
whole_data_id=$(basename $train_set)

mfccdir=mfcc

mkdir -p $dir

ref_rttm=$train_data_dir/rttm.annotation
if [ $stage -le 0 ]; then
  utils/copy_data_dir.sh data/train $train_data_dir
  cp data/train/rttm.annotation $ref_rttm 
fi

if [ $target_type == "forced" ]; then
  # Prepare forced alignments for the training data
  if [ $stage -le 1 ]; then
    steps/make_mfcc.sh --nj $nj --cmd "$train_cmd"  --write-utt2num-frames true \
      --mfcc-config conf/mfcc.conf $train_data_dir
    steps/compute_cmvn_stats.sh $train_data_dir
    utils/fix_data_dir.sh $train_data_dir
  fi

  if [ $stage -le 2 ]; then
    steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" \
      $train_data_dir data/lang $src_dir $ali_dir
  fi

  if [ $stage -le 3 ]; then
    steps/get_train_ctm.sh --use-segments false --stage 0 \
      $train_data_dir data/lang $ali_dir
  fi

  if [ $stage -le 4 ]; then
    local/generate_forced_aligned_rttm.py --max-pause 0.1 $ali_dir/ctm > $train_data_dir/rttm.forced
  fi

  ref_rttm=$train_data_dir/rttm.forced
fi

if [ $stage -le 5 ]; then
  # The training data may already be segmented, so we first prepare
  # a "whole" training data (not segmented) for training the overlap
  # detector.
  utils/data/convert_data_dir_to_whole.sh $train_data_dir $whole_data_dir
  steps/overlap/get_overlap_segments.py $ref_rttm > $whole_data_dir/overlap.rttm
fi

###############################################################################
# Extract features for the whole data directory. We extract 40-dim MFCCs to 
# train the NN-based overlap detector.
###############################################################################
if [ $stage -le 6 ]; then
  steps/make_mfcc.sh --nj $nj --cmd "$train_cmd"  --write-utt2num-frames true \
    --mfcc-config conf/mfcc_hires.conf ${whole_data_dir}
  steps/compute_cmvn_stats.sh ${whole_data_dir}
  utils/fix_data_dir.sh ${whole_data_dir}
fi

###############################################################################
# Prepare targets for training the overlap detector
###############################################################################
if [ $stage -le 7 ]; then
  steps/overlap/get_overlap_targets.py \
    ${whole_data_dir}/utt2num_frames ${whole_data_dir}/overlap.rttm - |\
    copy-feats ark,t:- ark,scp:$dir/targets.ark,$dir/targets.scp
fi

###############################################################################
# Train neural network for overlap detector
###############################################################################
if [ $stage -le 8 ]; then
  # Train a TDNN-LSTM network for SAD
  local/overlap/run_tdnn_lstm.sh \
    --targets-dir $dir --dir exp/overlap_$affix/tdnn_lstm \
    --data-dir ${whole_data_dir} || exit 1
fi

exit 0;
