#! /bin/bash

# Copyright 2017  Vimal Manohar
# Apache 2.0

# Features configs (Must match the features used to train the models
# $sat_model_dir and $model_dir)

lang=data/lang_nosp   # Must match the one used to train the models
lang_test=data/lang_nosp_sw1_tg  # Lang directory for decoding.

data_dir=data/train_100k_nodup
# Model directory used to align the $data_dir to get target labels for training
# SAD. This should typically be a speaker-adapted system.
sat_model_dir=exp/tri4
# Model direcotry used to decode the whole-recording version of the $data_dir to
# get target labels for training SAD. This should typically be a 
# speaker-independent system like LDA+MLLT system.
model_dir=exp/tri3
graph_dir=    # If not provided, a new one will be created using $lang_test

# Uniform segmentation options for decoding whole recordings. All values are in
# seconds.
max_segment_duration=10
overlap_duration=2.5
max_remaining_duration=5  # If the last remaining piece when splitting uniformly
                          # is smaller than this duration, then the last piece 
                          # is  merged with the previous.

# List of weights on labels obtained from alignment, 
# labels obtained from decoding and default labels in out-of-segment regions
merge_weights=1.0,0.1,0.5

prepare_targets_stage=-10
nstage=-10
train_stage=-10
test_stage=-10

affix=_1a
stage=-1
nj=80

. ./path.sh
. ./cmd.sh

set -e -u -o pipefail
. utils/parse_options.sh 

if [ $# -ne 0 ]; then
  exit 1
fi

dir=exp/segmentation${affix}
mkdir -p $dir

# See $lang/phones.txt and decide which should be garbage
garbage_phones="lau spn"
silence_phones="nsn SIL"

for p in $garbage_phones; do 
  for affix in "" "_B" "_E" "_I" "_S"; do
    echo "$p$affix"
  done
done > $dir/garbage_phones.txt

for p in $silence_phones; do 
  for affix in "" "_B" "_E" "_I" "_S"; do
    echo "$p$affix"
  done
done > $dir/silence_phones.txt

if ! cat $dir/garbage_phones.txt $dir/silence_phones.txt | \
  steps/segmentation/internal/verify_phones_list.py $lang/phones.txt; then
  echo "$0: Invalid $dir/{silence,garbage}_phones.txt"
  exit 1
fi

whole_data_dir=${data_dir}_whole

if [ $stage -le 0 ]; then
  utils/data/convert_data_dir_to_whole.sh $data_dir $whole_data_dir
fi

###############################################################################
# Extract features for the whole data directory
###############################################################################
if [ $stage -le 1 ]; then
  steps/make_mfcc.sh --cmd "$train_cmd" --nj $nj --write-utt2num-frames true \
    ${whole_data_dir} || exit 1
fi

###############################################################################
# Get feats for the manual segments
###############################################################################
if [ $stage -le 2 ]; then
  if [ ! -f ${data_dir}/segments ]; then
    utils/data/get_segments_for_data.sh $data_dir > $data_dir/segments
  fi
  utils/data/subsegment_data_dir.sh $whole_data_dir ${data_dir}/segments ${data_dir}/tmp
  cp $data_dir/tmp/feats.scp $data_dir
  awk '{print $1" "$2}' $data_dir/segments > $data_dir/utt2spk
  utils/utt2spk_to_spk2utt.pl $data_dir/utt2spk > $data_dir/spk2utt
fi

if [ $stage -le 3 ]; then
  steps/segmentation/prepare_targets_gmm.sh --stage $prepare_targets_stage \
    --train-cmd "$train_cmd" --decode-cmd "$decode_cmd" \
    --nj 80 --reco-nj 40 --lang-test $lang_test \
    --garbage-phones-list $dir/garbage_phones.txt \
    --silence-phones-list $dir/silence_phones.txt \
    $lang $data_dir $whole_data_dir $sat_model_dir $model_dir $dir
fi

if [ $stage -le 4 ]; then
  utils/copy_data_dir.sh ${whole_data_dir} ${whole_data_dir}_hires_bp
  steps/make_mfcc.sh --mfcc-config conf/mfcc_hires_bp.conf --nj 40 \
    ${whole_data_dir}_hires_bp
  steps/compute_cmvn_stats.sh ${whole_data_dir}_hires_bp
fi

if [ $stage -le 5 ]; then
  # Train a TDNN-LSTM network for SAD
  local/segmentation/tuning/train_lstm_asr_sad_1a.sh \
    --stage $nstage --train-stage $train_stage \
    --targets-dir $dir \
    --data-dir ${whole_data_dir}_hires_bp
fi

if [ $stage -le 6 ]; then
  # The options to this script must match the options used in the 
  # nnet training script. 
  # e.g. extra-left-context is 70, because the model is an LSTM trained with a 
  # chunk-left-context of 60. 
  # Note: frames-per-chunk is 150 even though the model was trained with 
  # chunk-width of 20. This is just for speed.
  # See the script for details of the options.
  steps/segmentation/detect_speech_activity.sh \
    --extra-left-context 70 --extra-right-context 0 --frames-per-chunk 150 \
    --extra-left-context-initial 0 --extra-right-context-final 0 \
    --nj 32 --acwt 0.3 --stage $test_stage \
    data/eval2000 \
    exp/segmentation_1a/tdnn_lstm_asr_sad_1a \
    mfcc_hires_bp \
    exp/segmentation_1a/tdnn_lstm_asr_sad_1a/{,eval2000}
fi
