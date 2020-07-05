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
test_nj=40
forced_alignments_file= # if provided, it will be used to obtain new segments
                        # and utt2spk files for the train and evaluation data

. ./cmd.sh

if [ -f ./path.sh ]; then . ./path.sh; fi

set -e -u -o pipefail
. utils/parse_options.sh 

if [ $# -ne 2 ] ; then
  echo >&2 "$0" "$@"
  echo >&2 "$0: Error: wrong number of arguments"
  echo -e >&2 "Usage:\n  $0 [opts] <corpus-dir> <librispeech-dir>"
  echo -e >&2 "eg:\n  $0 /export/corpora/LibriCSS /export/corpora/LibriSpeech"
  exit 1
fi

train_set=train_sim
test_sets="dev_sim test_sim" # These are the simulated LibriCSS test sets
dir=exp/overlap_${affix}

train_data_dir=data/${train_set}
whole_data_dir=data/${train_set}_whole
whole_data_id=$(basename $train_set)

mfccdir=mfcc

mkdir -p $dir

#################################################################
# DATA PREPARATION
#################################################################
if [ $stage -le 0 ]; then
  # First we prepare the simulated LibriCSS training data. We also
  # prepare development and evaluation sets. For each of these,
  # we additionally obtain force aligned segments to replace the
  # original segments. This will ensure that the system is evaluated
  # correctly, since the original segments may contain long silences.
  local/data_prep_sim.sh --forced-alignments $forced_alignments_file $1 $2
fi

if [ $stage -le 1 ]; then
  # Next prepare the overlap RTTM from the training data
  # This will contain segments with the labels: single, overlap
  local/overlap/get_overlap_segments.py ${train_data_dir}/rttm.forced |\
    awk '{if ($8!="overlap"){$8="single"}{print}}' > $train_data_dir/overlap_rttm
fi

if [ $stage -le 2 ]; then
  # The training data may already be segmented, so we first prepare
  # a "whole" training data (not segmented) for training the overlap
  # detector.
  utils/data/convert_data_dir_to_whole.sh $train_data_dir $whole_data_dir
  cp $train_data_dir/overlap_rttm $whole_data_dir/
fi

###############################################################################
# Extract features for the whole data directory. We extract 40-dim MFCCs to 
# train the NN-based overlap detector.
###############################################################################
if [ $stage -le 3 ]; then
  # spread the mfccs over various machines, as this data-set is quite large.
  if [[  $(hostname -f) ==  *.clsp.jhu.edu ]]; then
    mfcc=$(basename mfccdir) # in case was absolute pathname (unlikely), get basename.
    utils/create_split_dir.pl /export/fs{01,02,03}/$USER/kaldi-data/egs/libri_css/s5/$mfcc/storage \
     $mfccdir/storage
  fi
fi

if [ $stage -le 4 ]; then
  steps/make_mfcc.sh --nj $nj --cmd "$train_cmd"  --write-utt2num-frames true \
    --mfcc-config conf/mfcc_hires.conf \
    ${whole_data_dir} exp/make_mfcc/${whole_data_id} $mfccdir
  steps/compute_cmvn_stats.sh ${whole_data_dir} exp/make_mfcc/${whole_data_id} $mfccdir
  utils/fix_data_dir.sh ${whole_data_dir}
fi

###############################################################################
# Prepare targets for training the overlap detector
###############################################################################
if [ $stage -le 5 ]; then
  frame_shift=$( cat ${whole_data_dir}/frame_shift ) 
  local/overlap/get_overlap_targets.py \
    --frame-shift $frame_shift \
    ${whole_data_dir}/utt2num_frames $whole_data_dir/overlap_rttm - |\
    copy-feats ark,t:- ark,scp:$dir/targets.ark,$dir/targets.scp
fi

###############################################################################
# Train neural network for overlap detector
###############################################################################
if [ $stage -le 6 ]; then
  # Train a STATS-pooling network for SAD
  local/overlap/train_tdnn_1a.sh \
    --targets-dir $dir \
    --data-dir ${whole_data_dir} --affix "1a" || exit 1
fi

################################################################################
# Decoding on the simulated dev and test sets
################################################################################
if [ $stage -le 7 ]; then
  # First we prepare the reference overlap RTTM
  for dataset in $test_sets; do
    local/overlap/get_overlap_segments.py data/${dataset}/rttm.forced |\
      grep "overlap" > data/$dataset/overlap_rttm_ref
  done
fi

if [ $stage -le 8 ]; then
  # Next we extract features for decoding. If a segment file exists, we first move
  # it to .bak since we want to decode the whole recording.
  for dataset in $test_sets; do
    if [ -f data/$dataset/segments ]; then
      mv data/$dataset/segments data/$dataset/segments.bak
      mv data/$dataset/utt2spk data/$dataset/utt2spk.bak
      awk '{print $1, $1}' data/$dataset/wav.scp > data/$dataset/utt2spk
      utils/utt2spk_to_spk2utt.pl data/$dataset/utt2spk > data/$dataset/spk2utt
    fi
    steps/make_mfcc.sh --nj $test_nj --cmd "$train_cmd"  --write-utt2num-frames true \
      --mfcc-config conf/mfcc_hires.conf \
      data/$dataset exp/make_mfcc/$dataset $mfccdir
    steps/compute_cmvn_stats.sh data/$dataset exp/make_mfcc/$dataset $mfccdir
    utils/fix_data_dir.sh data/${dataset}
  done
fi

if [ $stage -le 9 ]; then
  # Finally we perform decoding with the overlap detector
  for dataset in $test_sets; do
    echo "$0: performing overlap detection on $dataset"
    local/detect_overlaps.sh data/$dataset --convert_data_dir_to_whole false \
      exp/overlap_1a/tdnn_stats_1a exp/overlap_1a/$dataset

    echo "$0: evaluating output.."
    md-eval.pl -r data/$dataset/overlap_rttm_ref -s exp/overlap_1a/$dataset/rttm_overlap |\
      awk 'or(/MISSED SPEECH/,/FALARM SPEECH/)'
  done
fi

exit 0;
