#!/usr/bin/env bash
# Copyright
#        2019   David Snyder
# Apache 2.0.
#
# This script is based on the run.sh script in the Voxceleb v2 recipe.
# It trains an x-vector DNN for diarization.

mfccdir=`pwd`/mfcc
vaddir=`pwd`/mfcc

voxceleb1_root=/export/corpora/VoxCeleb1
voxceleb2_root=/export/corpora/VoxCeleb2
data_dir=train_worn_simu_u400k
model_dir=exp/xvector_nnet_1a

stage=0
train_stage=-1

. ./cmd.sh

if [ -f ./path.sh ]; then . ./path.sh; fi
set -e -u -o pipefail
. utils/parse_options.sh

if [ $# -ne 0 ]; then
  exit 1
fi

if [ $stage -le 0 ]; then
  echo "$0: preparing voxceleb 2 data"
  local/make_voxceleb2.pl $voxceleb2_root dev data/voxceleb2_train
  local/make_voxceleb2.pl $voxceleb2_root test data/voxceleb2_test

  echo "$0: preparing voxceleb 1 data (see comments if this step fails)"
  # The format of the voxceleb 1 corpus has changed several times since it was
  # released.  Therefore, our dataprep scripts may or may not fail depending
  # on the version of the corpus you obtained.
  # If you downloaded the corpus soon after it was first released, this
  # version of the dataprep script might work:
  local/make_voxceleb1.pl $voxceleb1_root data/voxceleb1
  # However, if you've downloaded the corpus recently, you may need to use the
  # the following scripts instead:
  #local/make_voxceleb1_v2.pl $voxceleb1_root dev data/voxceleb1_train
  #local/make_voxceleb1_v2.pl $voxceleb1_root test data/voxceleb1_test

  # We should now have about 7,351 speakers and 1,277,503 utterances.
  utils/combine_data.sh data/voxceleb data/voxceleb2_train data/voxceleb2_test
fi

if [ $stage -le 1 ]; then
  echo "$0: preparing features for training data (voxceleb 1 + 2)"
  steps/make_mfcc.sh --write-utt2num-frames true \
    --mfcc-config conf/mfcc_hires.conf --nj 40 --cmd "$train_cmd" \
    data/voxceleb exp/make_mfcc $mfccdir
  utils/fix_data_dir.sh data/voxceleb
  # Note that we apply CMN to the MFCCs and write these to the disk.  These
  # features will later be used to train the x-vector DNN.
fi

# In this section, we augment the voxceleb data with reverberation.
# Note that we can probably improve the x-vector DNN if we include
# augmentations from the nonspeech regions of the Chime 6 training
# dataset.
if [ $stage -le 2 ]; then
  echo "$0: applying augmentation to x-vector training data (just reverb for now)"
  frame_shift=0.01
  awk -v frame_shift=$frame_shift '{print $1, $2*frame_shift;}' data/voxceleb/utt2num_frames > data/voxceleb/reco2dur

  if [ ! -d "RIRS_NOISES" ]; then
    echo "$0: downloading simulated room impulse response dataset"
    # Download the package that includes the real RIRs, simulated RIRs, isotropic noises and point-source noises
    wget --no-check-certificate http://www.openslr.org/resources/28/rirs_noises.zip
    unzip rirs_noises.zip
  fi

  # Make a version with reverberated speech
  rvb_opts=()
  rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/smallroom/rir_list")
  rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/mediumroom/rir_list")

  # Make a reverberated version of the training data.  Note that we don't add any
  # additive noise here.
  steps/data/reverberate_data_dir.py \
    "${rvb_opts[@]}" \
    --speech-rvb-probability 1 \
    --pointsource-noise-addition-probability 0 \
    --isotropic-noise-addition-probability 0 \
    --num-replications 1 \
    --source-sampling-rate 16000 \
    data/voxceleb data/voxceleb_reverb
  utils/copy_data_dir.sh --utt-suffix "-reverb" data/voxceleb_reverb data/voxceleb_reverb.new
  rm -rf data/voxceleb_reverb
  mv data/voxceleb_reverb.new data/voxceleb_reverb
fi

if [ $stage -le 3 ]; then
  echo "$0: making MFCCs for augmented training data"
  # Make MFCCs for the augmented data.  Note that we do not compute a new
  # vad.scp file here.  Instead, we use the vad.scp from the clean version of
  # the list.
  steps/make_mfcc.sh --mfcc-config conf/mfcc_hires.conf --nj 40 --cmd "$train_cmd" \
    data/voxceleb_reverb exp/make_mfcc $mfccdir
  # Combine the clean and augmented training data.  This is now roughly
  # double the size of the original clean list.
  utils/combine_data.sh data/voxceleb_combined data/voxceleb_reverb data/voxceleb
fi

# Now we prepare the features to generate examples for xvector training.
if [ $stage -le 4 ]; then
  # This script applies CMVN and removes nonspeech frames.  Note that this is somewhat
  # wasteful, as it roughly doubles the amount of training data on disk.  After
  # creating voxceleb examples, this can be removed.
  echo "$0: preparing features to train x-vector DNN"
  local/nnet3/xvector/prepare_feats.sh --nj 40 --cmd "$train_cmd" \
    data/voxceleb_combined data/voxceleb_combined_cmn exp/voxceleb_combined_cmn
  utils/fix_data_dir.sh data/voxceleb_combined_cmn
fi

if [ $stage -le 5 ]; then
  # Now, we need to remove features that are too short after removing silence
  # frames.  We want at least 4s (400 frames) per utterance.
  min_len=400
  mv data/voxceleb_combined_cmn/utt2num_frames data/voxceleb_combined_cmn/utt2num_frames.bak
  awk -v min_len=${min_len} '$2 > min_len {print $1, $2}' data/voxceleb_combined_cmn/utt2num_frames.bak > data/voxceleb_combined_cmn/utt2num_frames
  utils/filter_scp.pl data/voxceleb_combined_cmn/utt2num_frames data/voxceleb_combined_cmn/utt2spk > data/voxceleb_combined_cmn/utt2spk.new
  mv data/voxceleb_combined_cmn/utt2spk.new data/voxceleb_combined_cmn/utt2spk
  utils/fix_data_dir.sh data/voxceleb_combined_cmn

  # We also want several utterances per speaker. Now we'll throw out speakers
  # with fewer than 8 utterances.
  min_num_utts=8
  awk '{print $1, NF-1}' data/voxceleb_combined_cmn/spk2utt > data/voxceleb_combined_cmn/spk2num
  awk -v min_num_utts=${min_num_utts} '$2 >= min_num_utts {print $1, $2}' data/voxceleb_combined_cmn/spk2num | utils/filter_scp.pl - data/voxceleb_combined_cmn/spk2utt > data/voxceleb_combined_cmn/spk2utt.new
  mv data/voxceleb_combined_cmn/spk2utt.new data/voxceleb_combined_cmn/spk2utt
  utils/spk2utt_to_utt2spk.pl data/voxceleb_combined_cmn/spk2utt > data/voxceleb_combined_cmn/utt2spk

  utils/filter_scp.pl data/voxceleb_combined_cmn/utt2spk data/voxceleb_combined_cmn/utt2num_frames > data/voxceleb_combined_cmn/utt2num_frames.new
  mv data/voxceleb_combined_cmn/utt2num_frames.new data/voxceleb_combined_cmn/utt2num_frames

  utils/fix_data_dir.sh data/voxceleb_combined_cmn
fi

# Stages 6 through 8 are handled in run_xvector.sh.
# This script trains the x-vector DNN on the augmented voxceleb data.
local/nnet3/xvector/run_xvector.sh --stage $stage --train-stage $train_stage \
  --data data/voxceleb_combined_cmn --nnet-dir $model_dir \
  --egs-dir $model_dir/egs

if [ $stage -le 9 ]; then
  echo "$0: preparing a subset of Chime 6 training data to train PLDA model"
  utils/subset_data_dir.sh ${data_dir} 100000 data/plda_train
  steps/make_mfcc.sh --write-utt2num-frames true \
    --mfcc-config conf/mfcc_hires.conf --nj 40 --cmd "$train_cmd" \
    data/plda_train exp/make_mfcc $mfccdir
  utils/fix_data_dir.sh data/plda_train
  local/nnet3/xvector/prepare_feats.sh --nj 40 --cmd "$train_cmd" \
    data/plda_train data/plda_train_cmn exp/plda_train_cmn
  if [ -f data/plda_train/segments ]; then
    cp data/plda_train/segments data/plda_train_cmn/
  fi
fi

if [ $stage -le 10 ]; then
  echo "$0: extracting x-vector for PLDA training data"
  utils/fix_data_dir.sh data/plda_train_cmn
  diarization/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 10G" \
    --nj 40 --window 3.0 --period 10.0 --min-segment 1.5 --apply-cmn false \
    --hard-min true $model_dir \
    data/plda_train_cmn $model_dir/xvectors_plda_train
fi

# Train PLDA models
if [ $stage -le 11 ]; then
  echo "$0: training PLDA model"
  $train_cmd $model_dir/xvectors_plda_train/log/plda.log \
    ivector-compute-plda ark:$model_dir/xvectors_plda_train/spk2utt \
      "ark:ivector-subtract-global-mean \
      scp:$model_dir/xvectors_plda_train/xvector.scp ark:- \
      | transform-vec $model_dir/xvectors_plda_train/transform.mat ark:- ark:- \
      | ivector-normalize-length ark:- ark:- |" \
      $model_dir/xvectors_plda_train/plda || exit 1;
  cp $model_dir/xvectors_plda_train/plda $model_dir/
  cp $model_dir/xvectors_plda_train/transform.mat $model_dir/
  cp $model_dir/xvectors_plda_train/mean.vec $model_dir/
fi
