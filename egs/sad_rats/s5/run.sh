#!/usr/bin/env bash
# Copyright   2021   ARL (Author: John Morgan)
# Apache 2.0.

# This recipe builds a Speech Activity Detection system on the rats_sad corpus.
# The LDC catalog ID for the rats_sad corpus is LDC2015S02.
. ./cmd.sh
. ./path.sh
set -euo pipefail
stage=0

# Path where RATS_SAD gets downloaded (or where locally available):
if [[ $(hostname -f) == *.clsp.jhu.edu ]]; then
  rats_sad_data_dir=/export/corpora5/LDC/LDC2015S02/data
else
  rats_sad_data_dir=/mnt/corpora/LDC2015S02/RATS_SAD/data
fi
nj=50
test_sets="dev-1 dev-2 "
affix=1a
dir=exp/sad_${affix}

. utils/parse_options.sh

if [ $stage -le 0 ]; then
  # Consolidate the data from the corpus annotation files
  echo "$0: Get  all info from annotation files."
  local/rats_sad_tab_prep.sh $rats_sad_data_dir
fi

if [ $stage -le 1 ]; then
  # Prepare data directories
  for fld in train $test_sets ; do
    echo "$0: preparing $fld data set."
    mkdir -p data/$fld
    local/prepare_data.py data/local/annotations/$fld.txt \
      $rats_sad_data_dir/$fld/audio/ data/$fld
  done
fi
exit

if [ $stage -le 2 ]; then
  # Write utt2spk and segments files
  for fld in train $test_sets ; do
    echo "$0: Convert $fld rttm files to utt2spk and segments for ${fld}."
    local/convert_rttm_to_utt2spk_and_segments.py --use-reco-id-as-spkr=true \
      data/$fld/rttm.annotation \
      <(awk '{print $2" "$2" "$3}' data/$fld/rttm.annotation |sort -u) \
      data/$fld/utt2spk data/$fld/segments

    utils/utt2spk_to_spk2utt.pl data/$fld/utt2spk > data/$fld/spk2utt
    utils/fix_data_dir.sh data/$fld
  done
fi

if [ $stage -le 3 ]; then
  # Get supervision for whole recordings from segments supervision
  echo "$0: Prepare a 'whole' training data (not segmented) for training the SAD."
  utils/copy_data_dir.sh data/train data/train_sad
  cp data/train/rttm.annotation data/train_sad
  utils/data/convert_data_dir_to_whole.sh data/train_sad data/train_sad_whole
fi

if [ $stage -le 4 ]; then
  # extract MFCCs for whole recordings
  echo "$0: Extract features for the 'whole' data directory."
  steps/make_mfcc.sh --nj $nj --cmd "$train_cmd"  --write-utt2num-frames true \
    --mfcc-config conf/mfcc_hires.conf data/train_sad_whole
  steps/compute_cmvn_stats.sh data/train_sad_whole
  utils/fix_data_dir.sh data/train_sad_whole
fi

if [ $stage -le 5 ]; then
  # Associate silence or speech labels to frames
  echo "$0: Get targets for training data set."
  mkdir -p $dir
  local/get_speech_targets.py \
    data/train_sad_whole/utt2num_frames \
    data/train_sad/rttm.annotation - |\
    copy-feats ark:- ark,scp:$dir/targets.ark,$dir/targets.scp
fi

if [ $stage -le 6 ]; then
  # Train the SAD neural network model
  echo "$0: Train a TDNN+LSTM network for SAD."
  local/segmentation/run_lstm.sh \
    --stage 0 --train-stage -10 \
    --targets-dir $dir \
    --data-dir data/train_sad_whole --affix $affix || exit 1
fi

if [ $stage -le 7 ]; then
  # Run SAD on test sets
  for fld in $test_sets; do
    echo "$0: Run SAD detection on $fld."
    local/detect_speech_activity.sh $fld
  done
fi

if [ $stage -le 8 ]; then
  # Write rttm files
  for fld in $test_sets; do
    echo "$0: Writing rttm file for $fld."
    steps/segmentation/convert_utt2spk_and_segments_to_rttm.py \
      data/${fld}_seg/utt2spk \
      $dir/$fld/segments \
    $dir/$fld/sad.rttm
  done
fi

if [ $stage -le 9 ]; then
  # Evaluate SAD output
  for fld in $test_sets; do
    echo "$0: evaluating $fld output."
    md-eval.pl -c 0 -r data/$fld/rttm.annotation \
      -s $dir/$fld/sad.rttm | awk '/(MISSED|FALARM) SPEECH/' > \
      $dir/$fld/results.txt
  done
fi
