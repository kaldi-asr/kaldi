#!/bin/bash
#
# Apache 2.0.
#
# See README.txt for more info on data required.
# Results (EERs) are inline in comments below.

. cmd.sh
. path.sh
set -e
fbankdir=`pwd`/fbank
vaddir=`pwd`/fbank

# SRE16 trials
sre16_trials=data/sre16_eval_test/trials
sre16_trials_tgl=data/sre16_eval_test/trials_tgl
sre16_trials_yue=data/sre16_eval_test/trials_yue
nnet_dir=exp/xvector_nnet_1a

stage=6
if [ $stage -le 0 ]; then
  # Path to some, but not all of the training corpora
  data_root=/export/corpora/LDC

  # Prepare telephone and microphone speech from Mixer6.
  local/make_mx6.sh $data_root/LDC2013S03 data/

  # Prepare SRE10 test and enroll. Includes microphone interview speech.
  # NOTE: This corpus is now available through the LDC as LDC2017S06.
  local/make_sre10.pl /export/corpora5/SRE/SRE2010/eval/ data/

  # Prepare SRE08 test and enroll. Includes some microphone speech.
  local/make_sre08.pl $data_root/LDC2011S08 $data_root/LDC2011S05 data/

  # This prepares the older NIST SREs from 2004-2006.
  local/make_sre.sh $data_root data/

  # Combine all SREs prior to 2016 and Mixer6 into one dataset
  utils/combine_data.sh data/sre \
    data/sre2004 data/sre2005_train \
    data/sre2005_test data/sre2006_train \
    data/sre2006_test_1 data/sre2006_test_2 \
    data/sre08 data/mx6 data/sre10
  utils/validate_data_dir.sh --no-text --no-feats data/sre
  utils/fix_data_dir.sh data/sre

  # Prepare SWBD corpora.
  local/make_swbd_cellular1.pl $data_root/LDC2001S13 \
    data/swbd_cellular1_train
  local/make_swbd_cellular2.pl /export/corpora5/LDC/LDC2004S07 \
    data/swbd_cellular2_train
  local/make_swbd2_phase1.pl $data_root/LDC98S75 \
    data/swbd2_phase1_train
  local/make_swbd2_phase2.pl /export/corpora5/LDC/LDC99S79 \
    data/swbd2_phase2_train
  local/make_swbd2_phase3.pl /export/corpora5/LDC/LDC2002S06 \
    data/swbd2_phase3_train

  # Combine all SWB corpora into one dataset.
  utils/combine_data.sh data/swbd \
    data/swbd_cellular1_train data/swbd_cellular2_train \
    data/swbd2_phase1_train data/swbd2_phase2_train data/swbd2_phase3_train

  # Prepare NIST SRE 2016 evaluation data.
  local/make_sre16_eval.pl /export/corpora5/SRE/R149_0_1 data

  # Prepare unlabeled Cantonese and Tagalog development data. This dataset
  # was distributed to SRE participants.
  local/make_sre16_unlabeled.pl /export/corpora5/SRE/LDC2016E46_SRE16_Call_My_Net_Training_Data data
fi

if [ $stage -le 1 ]; then
  # Make MFCCs and compute the energy-based VAD for each dataset
  for name in sre swbd sre16_eval_enroll sre16_eval_test sre16_major; do
    steps/make_fbank.sh --fbank-config conf/fbank.conf --nj 40 --cmd "$train_cmd" \
      data/${name} exp/make_fbank $fbankdir
    utils/fix_data_dir.sh data/${name}
    sid/compute_vad_decision.sh --nj 40 --cmd "$train_cmd" \
      data/${name} exp/make_vad $vaddir
    utils/fix_data_dir.sh data/${name}
  done
  utils/combine_data.sh data/swbd_sre data/swbd data/sre
  utils/fix_data_dir.sh data/swbd_sre
fi

# In this section, we augment the SWBD and SRE data with reverberation,
# noise, music, and babble, and combined it with the clean data.
# The combined list will be used to train the xvector DNN.  The SRE
# subset will be used to train the PLDA model.
if [ $stage -le 2 ]; then
  utils/data/get_utt2num_frames.sh --nj 40 --cmd "$train_cmd" data/swbd_sre
  frame_shift=0.01
  awk -v frame_shift=$frame_shift '{print $1, $2*frame_shift;}' data/swbd_sre/utt2num_frames > data/swbd_sre/reco2dur

  if [ ! -d "RIRS_NOISES" ]; then
    # Download the package that includes the real RIRs, simulated RIRs, isotropic noises and point-source noises
    wget --no-check-certificate http://www.openslr.org/resources/28/rirs_noises.zip
    unzip rirs_noises.zip
  fi

  # Make a version with reverberated speech
  rvb_opts=()
  rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/smallroom/rir_list")
  rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/mediumroom/rir_list")

  # Make a reverberated version of the SWBD+SRE list.  Note that we don't add any
  # additive noise here.
  python steps/data/reverberate_data_dir.py \
    "${rvb_opts[@]}" \
    --speech-rvb-probability 1 \
    --pointsource-noise-addition-probability 0 \
    --isotropic-noise-addition-probability 0 \
    --num-replications 1 \
    --source-sampling-rate 8000 \
    data/swbd_sre data/swbd_sre_reverb
  cp data/swbd_sre/vad.scp data/swbd_sre_reverb/
  utils/copy_data_dir.sh --utt-suffix "-reverb" data/swbd_sre_reverb data/swbd_sre_reverb.new
  rm -rf data/swbd_sre_reverb
  mv data/swbd_sre_reverb.new data/swbd_sre_reverb

  # Prepare the MUSAN corpus, which consists of music, speech, and noise
  # suitable for augmentation.
  local/make_musan.sh /export/corpora/JHU/musan data

  # Get the duration of the MUSAN recordings.  This will be used by the
  # script augment_data_dir.py.
  for name in speech noise music; do
    utils/data/get_utt2dur.sh data/musan_${name}
    mv data/musan_${name}/utt2dur data/musan_${name}/reco2dur
  done

  # Augment with musan_noise
  python steps/data/augment_data_dir.py --utt-suffix "noise" --fg-interval 1 --fg-snrs "15:10:5:0" --fg-noise-dir "data/musan_noise" data/swbd_sre data/swbd_sre_noise
  # Augment with musan_music
  python steps/data/augment_data_dir.py --utt-suffix "music" --bg-snrs "15:10:8:5" --num-bg-noises "1" --bg-noise-dir "data/musan_music" data/swbd_sre data/swbd_sre_music
  # Augment with musan_speech
  python steps/data/augment_data_dir.py --utt-suffix "babble" --bg-snrs "20:17:15:13" --num-bg-noises "3:4:5:6:7" --bg-noise-dir "data/musan_speech" data/swbd_sre data/swbd_sre_babble

  # Combine reverb, noise, music, and babble into one directory.
  utils/combine_data.sh data/swbd_sre_aug data/swbd_sre_reverb data/swbd_sre_noise data/swbd_sre_music data/swbd_sre_babble

  # Take a random subset of the augmentations (128k is somewhat larger than twice
  # the size of the SWBD+SRE list)
  utils/subset_data_dir.sh data/swbd_sre_aug 128000 data/swbd_sre_aug_128k
  utils/fix_data_dir.sh data/swbd_sre_aug_128k

  # Make filterbanks for the augmented data.  Note that we do not compute a new
  # vad.scp file here.  Instead, we use the vad.scp from the clean version of
  # the list.
  steps/make_fbank.sh --fbank-config conf/fbank.conf --nj 40 --cmd "$train_cmd" \
    data/swbd_sre_aug_128k exp/make_fbank $fbankdir

  # Combine the clean and augmented SWBD+SRE list.  This is now roughly
  # double the size of the original clean list.
  utils/combine_data.sh data/swbd_sre_combined data/swbd_sre_aug_128k data/swbd_sre
fi

# Now we compute the features needed for xvector training.  This consists of
# applying sliding window CMN and removing silence frames.
if [ $stage -le 3 ]; then
  local/nnet3/xvector/prepare_xvector_feats_cmvn_remove_sil.sh --nj 40 --cmd "$train_cmd" \
    data/swbd_sre_combined data/swbd_sre_combined_no_sil exp/swbd_sre_combined_no_sil
  utils/fix_data_dir.sh data/swbd_sre_combined_no_sil
  utils/data/get_utt2num_frames.sh --nj 40 --cmd "$train_cmd" data/swbd_sre_combined_no_sil
  utils/fix_data_dir.sh data/swbd_sre_combined_no_sil
  # Now, we need to remove features that are too short after removing silence
  # frames.  We want atleast 5s (500 frames) per utterance.
  min_len=500
  mv data/swbd_sre_combined_no_sil/utt2num_frames data/swbd_sre_combined_no_sil/utt2num_frames.bak
  awk -v min_len=${min_len} '$2 > min_len {print $1, $2}' data/swbd_sre_combined_no_sil/utt2num_frames.bak > data/swbd_sre_combined_no_sil/utt2num_frames
  utils/filter_scp.pl data/swbd_sre_combined_no_sil/utt2num_frames data/swbd_sre_combined_no_sil/utt2spk > data/swbd_sre_combined_no_sil/utt2spk.new
  mv data/swbd_sre_combined_no_sil/utt2spk.new data/swbd_sre_combined_no_sil/utt2spk
  utils/fix_data_dir.sh data/swbd_sre_combined_no_sil

  # We also want several utterances per speaker. Now we'll throw out speakers
  # with fewer than 8 utterances.
  min_num_utts=8
  awk '{print $1, NF-1}' data/swbd_sre_combined_no_sil/spk2utt > data/swbd_sre_combined_no_sil/spk2num
  awk -v min_num_utts=${min_num_utts} '$2 >= min_num_utts {print $1, $2}' data/swbd_sre_combined_no_sil/spk2num | utils/filter_scp.pl - data/swbd_sre_combined_no_sil/spk2utt > data/swbd_sre_combined_no_sil/spk2utt.new
  mv data/swbd_sre_combined_no_sil/spk2utt.new data/swbd_sre_combined_no_sil/spk2utt
  utils/spk2utt_to_utt2spk.pl data/swbd_sre_combined_no_sil/spk2utt > data/swbd_sre_combined_no_sil/utt2spk

  utils/filter_scp.pl data/swbd_sre_combined_no_sil/utt2spk data/swbd_sre_combined_no_sil/utt2num_frames > data/swbd_sre_combined_no_sil/utt2num_frames.new
  mv data/swbd_sre_combined_no_sil/utt2num_frames.new data/swbd_sre_combined_no_sil/utt2num_frames

  # Now we're reaady to create training examples.
  utils/fix_data_dir.sh data/swbd_sre_combined_no_sil
fi

local/nnet3/xvector/run_xvector.sh --stage $stage --train-stage 16 \
  --data data/swbd_sre_combined_no_sil --nnet-dir $nnet_dir \
  --egs-dir $nnet_dir/egs
