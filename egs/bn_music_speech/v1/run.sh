#!/usr/bin/env bash
# Copyright 2015   David Snyder
# Apache 2.0.
#
# This example demonstrates music/speech discrimination. This recipe trains
# three GMMs on the music, speech and noise portions of the MUSAN corpus.
# We test the systems on Broadcast News. The Broadcast News test data consists
# of short segments of either speech or music. The classification decisions
# are made at a segment level from the average likelihoods of two GMMs.
# Results (EERs) are inline in comments below.
#
# See README.txt for more info on data required.

. ./cmd.sh
. ./path.sh
set -e
mfccdir=`pwd`/mfcc
vaddir=`pwd`/mfcc

local/make_bn.sh /export/corpora5/LDC/LDC97S44 \
                 /export/corpora/LDC/LDC97T22 data

steps/data/make_musan.sh --sampling-rate 16000 /export/corpora/JHU/musan data

steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 30 --cmd "$train_cmd" \
    data/musan_speech exp/make_mfcc $mfccdir
steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 30 --cmd "$train_cmd" \
    data/musan_music exp/make_mfcc $mfccdir
steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 5 --cmd "$train_cmd" \
    data/musan_noise exp/make_mfcc $mfccdir
steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 30 --cmd "$train_cmd" \
    data/bn exp/make_mfcc $mfccdir

utils/fix_data_dir.sh data/musan_speech
utils/fix_data_dir.sh data/musan_music
utils/fix_data_dir.sh data/musan_noise
utils/fix_data_dir.sh data/bn

sid/compute_vad_decision.sh --nj 20 --cmd "$train_cmd" \
    data/musan_speech exp/make_vad $vaddir
sid/compute_vad_decision.sh --nj 5 --cmd "$train_cmd" \
    data/musan_noise exp/make_vad $vaddir
sid/compute_vad_decision.sh --nj 20 --cmd "$train_cmd" \
    data/musan_music exp/make_vad $vaddir
sid/compute_vad_decision.sh --nj 20 --cmd "$train_cmd" \
    data/bn exp/make_vad $vaddir

sid/train_diag_ubm.sh --nj 10 --cmd "$train_cmd" --delta-window 2 \
    data/musan_noise 32 exp/diag_ubm_noise &
sid/train_diag_ubm.sh --nj 20 --cmd "$train_cmd" --delta-window 2 \
    data/musan_speech 32 exp/diag_ubm_speech &
sid/train_diag_ubm.sh --nj 20 --cmd "$train_cmd" --delta-window 2 \
    data/musan_music 32  exp/diag_ubm_music
wait;

sid/train_full_ubm.sh --nj 20 --cmd "$train_cmd" \
    --remove-low-count-gaussians false data/musan_noise \
    exp/diag_ubm_noise exp/full_ubm_noise &
sid/train_full_ubm.sh --nj 20 --cmd "$train_cmd" \
    --remove-low-count-gaussians false data/musan_speech \
    exp/diag_ubm_speech exp/full_ubm_speech &
sid/train_full_ubm.sh --nj 20 --cmd "$train_cmd" \
    --remove-low-count-gaussians false data/musan_music \
    exp/diag_ubm_music exp/full_ubm_music
wait;

sid/music_id.sh --cmd "$train_cmd" --nj 40 \
  exp/full_ubm_music exp/full_ubm_speech \
  data/bn exp/bn_music_speech
sid/music_id.sh --cmd "$train_cmd" --nj 40 \
  exp/full_ubm_noise exp/full_ubm_speech \
  data/bn exp/bn_noise_speech

printf "EER using GMMs trained on music and speech"
compute-eer <(local/print_scores.py exp/bn_music_speech/ratio)
# Equal error rate is 0.344234%, at threshold 0.525752
printf "\nEER using GMM trained on noise instead of music"
compute-eer <(local/print_scores.py exp/bn_noise_speech/ratio)
# Equal error rate is 0.860585%, at threshold 0.123218

# The following script replaces the VAD decisions originally computed by
# the energy-based VAD.  It uses the GMMs trained earlier in the script
# to make frame-level decisions. Due to the mapping provided in
# conf/merge_vad_map.txt, "0" corresponds to silence, "1" to speech, and
# "2" to music.
sid/compute_vad_decision_gmm.sh --nj 40 --cmd "$train_cmd" \
  --merge-map-config conf/merge_vad_map.txt --use-energy-vad true \
  data/bn exp/full_ubm_noise exp/full_ubm_speech/ \
  exp/full_ubm_music/ exp/vad_gmm exp/vad_gmm/
