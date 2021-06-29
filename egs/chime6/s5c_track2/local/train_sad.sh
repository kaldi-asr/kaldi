#!/usr/bin/env bash

# Copyright  2017  Nagendra Kumar Goel
#            2017  Vimal Manohar
#            2019  Desh Raj
# Apache 2.0

# This script is based on local/run_asr_segmentation.sh script in the
# Aspire recipe. It demonstrates nnet3-based speech activity detection for
# segmentation.
# This script:
# 1) Prepares targets (per-frame labels) for a subset of training data 
#    using GMM models
# 2) Trains TDNN+Stats or TDNN+LSTM neural network using the targets 
# 3) Demonstrates using the SAD system to get segments of dev data

lang=data/lang   # Must match the one used to train the models
lang_test=data/lang_test  # Lang directory for decoding.

data_dir=
test_sets=
# Model directory used to align the $data_dir to get target labels for training
# SAD. This should typically be a speaker-adapted system.
sat_model_dir=
# Model direcotry used to decode the whole-recording version of the $data_dir to
# get target labels for training SAD. This should typically be a
# speaker-independent system like LDA+MLLT system.
model_dir=
graph_dir=                  # Graph for decoding whole-recording version of $data_dir.
                            # If not provided, a new one will be created using $lang_test

# List of weights on labels obtained from alignment;
# labels obtained from decoding; and default labels in out-of-segment regions
merge_weights=1.0,0.1,0.5

prepare_targets_stage=-10
nstage=-10
train_stage=-10
stage=0
nj=50
reco_nj=40

# test options
test_nj=10

. ./cmd.sh
. ./conf/sad.conf

if [ -f ./path.sh ]; then . ./path.sh; fi

set -e -u -o pipefail
. utils/parse_options.sh 

if [ $# -ne 0 ]; then
  exit 1
fi

dir=exp/segmentation${affix}
sad_work_dir=exp/sad${affix}_${nnet_type}/
sad_nnet_dir=$dir/tdnn_${nnet_type}_sad_1a

mkdir -p $dir
mkdir -p ${sad_work_dir}

# See $lang/phones.txt and decide which should be garbage
garbage_phones="laughs inaudible"
silence_phones="sil spn noise"

for p in $garbage_phones; do 
  for a in "" "_B" "_E" "_I" "_S"; do
    echo "$p$a"
  done
done > $dir/garbage_phones.txt

for p in $silence_phones; do 
  for a in "" "_B" "_E" "_I" "_S"; do
    echo "$p$a"
  done
done > $dir/silence_phones.txt

if ! cat $dir/garbage_phones.txt $dir/silence_phones.txt | \
  steps/segmentation/internal/verify_phones_list.py $lang/phones.txt; then
  echo "$0: Invalid $dir/{silence,garbage}_phones.txt"
  exit 1
fi

# The training data may already be segmented, so we first prepare
# a "whole" training data (not segmented) for training the SAD
# system.

whole_data_dir=${data_dir}_whole
whole_data_id=$(basename $whole_data_dir)

if [ $stage -le 0 ]; then
  utils/data/convert_data_dir_to_whole.sh $data_dir $whole_data_dir
fi

###############################################################################
# Extract features for the whole data directory. We extract 13-dim MFCCs to
# generate targets using the GMM system, and 40-dim MFCCs to train the NN-based
# SAD.
###############################################################################
if [ $stage -le 1 ]; then
  steps/make_mfcc.sh --nj $reco_nj --cmd "$train_cmd"  --write-utt2num-frames true \
    --mfcc-config conf/mfcc.conf \
    $whole_data_dir exp/make_mfcc/${whole_data_id}
  steps/compute_cmvn_stats.sh $whole_data_dir exp/make_mfcc/${whole_data_id}
  utils/fix_data_dir.sh $whole_data_dir

  utils/copy_data_dir.sh $whole_data_dir ${whole_data_dir}_hires
  steps/make_mfcc.sh --nj $reco_nj --cmd "$train_cmd"  --write-utt2num-frames true \
    --mfcc-config conf/mfcc_hires.conf \
    ${whole_data_dir}_hires exp/make_mfcc/${whole_data_id}_hires
  steps/compute_cmvn_stats.sh ${whole_data_dir}_hires exp/make_mfcc/${whole_data_id}_hires
  utils/fix_data_dir.sh ${whole_data_dir}_hires
fi

###############################################################################
# Prepare SAD targets for recordings
###############################################################################
targets_dir=$dir/${whole_data_id}_combined_targets_sub3
if [ $stage -le 2 ]; then
  steps/segmentation/prepare_targets_gmm.sh --stage $prepare_targets_stage \
    --train-cmd "$train_cmd" --decode-cmd "$decode_cmd" \
    --nj $nj --reco-nj $reco_nj --lang-test $lang \
    --garbage-phones-list $dir/garbage_phones.txt \
    --silence-phones-list $dir/silence_phones.txt \
    --merge-weights "$merge_weights" \
    --remove-mismatch-frames false \
    --graph-dir "$graph_dir" \
    $lang $data_dir $whole_data_dir $sat_model_dir $model_dir $dir
fi

###############################################################################
# Train a neural network for SAD
###############################################################################
if [ $stage -le 3 ]; then
	if [ $nnet_type == "stats" ]; then
		# Train a STATS-pooling network for SAD
		local/segmentation/tuning/train_stats_sad_1a.sh \
		  --stage $nstage --train-stage $train_stage \
		  --targets-dir ${targets_dir} \
		  --data-dir ${whole_data_dir}_hires --affix "1a" || exit 1
	
	elif [ $nnet_type == "lstm" ]; then
    # Train a TDNN+LSTM network for SAD
    local/segmentation/tuning/train_lstm_sad_1a.sh \
      --stage $nstage --train-stage $train_stage \
      --targets-dir ${targets_dir} \
      --data-dir ${whole_data_dir}_hires --affix "1a" || exit 1

  fi
fi

exit 0;
