#!/bin/bash

# Copyright  2017  Nagendra Kumar Goel
#            2017  Vimal Manohar
# Apache 2.0

# We assume the run.sh has been executed (because we are using model
# directories like exp/tri4)

# This script demonstrates nnet3-based speech activity detection for
# segmentation.
# This script:
# 1) Prepares targets (per-frame labels) for a subset of training data 
#    using GMM models
# 2) Augments the training data with reverberation and additive noise
# 3) Trains TDNN+Stats or TDNN+LSTM neural network using the targets 
#    and augmented data
# 4) Demonstrates using the SAD system to get segments of eval data and decode

lang=data/lang   # Must match the one used to train the models
lang_test=data/lang_nosp_sw1_tg  # Lang directory for decoding.

data_dir=data/train_nodup
# Model directory used to align the $data_dir to get target labels for training
# SAD. This should typically be a speaker-adapted system.
sat_model_dir=exp/tri4
# Model direcotry used to decode the whole-recording version of the $data_dir to
# get target labels for training SAD. This should typically be a 
# speaker-independent system like LDA+MLLT system.
model_dir=exp/tri3
graph_dir=    # Graph for decoding whole-recording version of $data_dir.
              # If not provided, a new one will be created using $lang_test

# List of weights on labels obtained from alignment;
# labels obtained from decoding; and default labels in out-of-segment regions
merge_weights=1.0,0.1,0.5

prepare_targets_stage=-10
nstage=-10
train_stage=-10
num_data_reps=2
affix=_1a   # For segmentation
stage=-1
nj=80
reco_nj=40

# test options
test_stage=-10
test_nj=32

. ./cmd.sh
if [ -f ./path.sh ]; then . ./path.sh; fi

set -e -u -o pipefail
. utils/parse_options.sh 

if [ $# -ne 0 ]; then
  exit 1
fi

dir=exp/segmentation${affix}
mkdir -p $dir

# See $lang/phones.txt and decide which should be garbage
garbage_phones="lau spn"
silence_phones="sil"

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

whole_data_dir=${data_dir}_whole
whole_data_id=$(basename $whole_data_dir)

rvb_data_dir=${whole_data_dir}_rvb_hires

if [ $stage -le 0 ]; then
  utils/data/convert_data_dir_to_whole.sh $data_dir $whole_data_dir
fi

###############################################################################
# Extract features for the whole data directory
###############################################################################
if [ $stage -le 1 ]; then
  steps/make_mfcc.sh --nj $reco_nj --cmd "$train_cmd"  --write-utt2num-frames true \
    $whole_data_dir exp/make_mfcc/${whole_data_id}
  steps/compute_cmvn_stats.sh $whole_data_dir exp/make_mfcc/${whole_data_id}
  utils/fix_data_dir.sh $whole_data_dir
fi

###############################################################################
# Prepare SAD targets for recordings
###############################################################################
targets_dir=$dir/${whole_data_id}_combined_targets_sub3
if [ $stage -le 3 ]; then
  steps/segmentation/prepare_targets_gmm.sh --stage $prepare_targets_stage \
    --train-cmd "$train_cmd" --decode-cmd "$decode_cmd" \
    --nj $nj --reco-nj $reco_nj --lang-test $lang_test \
    --garbage-phones-list $dir/garbage_phones.txt \
    --silence-phones-list $dir/silence_phones.txt \
    --merge-weights "$merge_weights" \
    --graph-dir "$graph_dir" \
    $lang $data_dir $whole_data_dir $sat_model_dir $model_dir $dir
fi

rvb_targets_dir=${targets_dir}_rvb
if [ $stage -le 4 ]; then
  # Download the package that includes the real RIRs, simulated RIRs, isotropic noises and point-source noises
  if [ ! -f rirs_noises.zip ]; then
    wget --no-check-certificate http://www.openslr.org/resources/28/rirs_noises.zip
    unzip rirs_noises.zip
  fi

  rvb_opts=()
  # This is the config for the system using simulated RIRs and point-source noises
  rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/smallroom/rir_list")
  rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/mediumroom/rir_list")
  rvb_opts+=(--noise-set-parameters RIRS_NOISES/pointsource_noises/noise_list)

  foreground_snrs="20:10:15:5:0"
  background_snrs="20:10:15:5:0"
  # corrupt the data to generate multi-condition data
  # for data_dir in train dev test; do
  python steps/data/reverberate_data_dir.py \
    "${rvb_opts[@]}" \
    --prefix "rev" \
    --foreground-snrs $foreground_snrs \
    --background-snrs $background_snrs \
    --speech-rvb-probability 0.5 \
    --pointsource-noise-addition-probability 0.5 \
    --isotropic-noise-addition-probability 0.7 \
    --num-replications $num_data_reps \
    --max-noises-per-minute 4 \
    --source-sampling-rate 8000 \
    $whole_data_dir $rvb_data_dir
fi

if [ $stage -le 5 ]; then
  steps/make_mfcc.sh --mfcc-config conf/mfcc_hires.conf --nj $reco_nj \
    ${rvb_data_dir}
  steps/compute_cmvn_stats.sh ${rvb_data_dir}
  utils/fix_data_dir.sh $rvb_data_dir
fi

if [ $stage -le 6 ]; then
  rvb_targets_dirs=()
  for i in `seq 1 $num_data_reps`; do
    steps/segmentation/copy_targets_dir.sh --utt-prefix "rev${i}_" \
      $targets_dir ${targets_dir}_temp_$i || exit 1
    rvb_targets_dirs+=(${targets_dir}_temp_$i)
  done

  steps/segmentation/combine_targets_dirs.sh \
    $rvb_data_dir ${rvb_targets_dir} \
    ${rvb_targets_dirs[@]} || exit 1;

  rm -r ${rvb_targets_dirs[@]}
fi

if [ $stage -le 7 ]; then
  # Train a STATS-pooling network for SAD
  local/segmentation/tuning/train_stats_asr_sad_1a.sh \
    --stage $nstage --train-stage $train_stage \
    --targets-dir ${rvb_targets_dir} \
    --data-dir ${rvb_data_dir} --affix "1a" || exit 1
fi

if [ $stage -le 8 ]; then
  # The options to this script must match the options used in the 
  # nnet training script. 
  # e.g. extra-left-context is 79, because the model is an stats pooling network 
  # trained with a chunk-left-context of 79 and chunk-right-context of 21. 
  # Note: frames-per-chunk is 150 even though the model was trained with 
  # chunk-width of 20. This is just for speed.
  # See the script for details of the options.
  steps/segmentation/detect_speech_activity.sh \
    --extra-left-context 79 --extra-right-context 21 --frames-per-chunk 150 \
    --extra-left-context-initial 0 --extra-right-context-final 0 \
    --nj $test_nj --acwt 0.3 --stage $test_stage \
    data/eval2000 \
    exp/segmentation${affix}/tdnn_stats_asr_sad_1a \
    mfcc_hires \
    exp/segmentation${affix}/tdnn_stats_asr_sad_1a/{,eval2000}
fi

if [ $stage -le 9 ]; then
  # Do some diagnostics
  steps/segmentation/evaluate_segmentation.pl data/eval2000/segments \
    exp/segmentation${affix}/tdnn_stats_asr_sad_1a/eval2000_seg/segments &> \
    exp/segmentation${affix}/tdnn_stats_asr_sad_1a/eval2000_seg/evalutate_segmentation.log
  
  steps/segmentation/convert_utt2spk_and_segments_to_rttm.py \
    exp/segmentation${affix}/tdnn_stats_asr_sad_1a/eval2000_seg/utt2spk \
    exp/segmentation${affix}/tdnn_stats_asr_sad_1a/eval2000_seg/segments \
    exp/segmentation${affix}/tdnn_stats_asr_sad_1a/eval2000_seg/sys.rttm

#  export PATH=$PATH:$KALDI_ROOT/tools/sctk/bin
#  md-eval.pl -c 0.25 -r $eval2000_rttm_file \
#    -s exp/segmentation${affix}/tdnn_stats_asr_sad_1a/eval2000_seg/sys.rttm > \
#    exp/segmentation${affix}/tdnn_stats_asr_sad_1a/eval2000_seg/md_eval.log
fi

if [ $stage -le 10 ]; then
  utils/copy_data_dir.sh exp/segmentation${affix}/tdnn_stats_asr_sad_1a/eval2000_seg \
    data/eval2000.seg_asr_sad_1a
fi
