#!/bin/bash
# Copyright 2016  David Snyder
# TODO
# Apache 2.0.
#
# TODO details on what this does.
# See README for more info on the required data.

. cmd.sh
. path.sh
set -e
mfccdir=`pwd`/mfcc
vaddir=`pwd`/mfcc
num_components=2048
ivector_dim=128

# Prepare a collection of NIST SRE data.
# TODO: This will probably be useful for UBM, ivector extractor training, and possibly, PLDA
#
local/make_sre.sh data

# Prepare SWB for UBM and i-vector extractor training.
# TODO: This is probably reasonable training data for the Callhome system, but it might also
#       be a good idea to try Fisher.
local/make_swbd2_phase2.pl /export/corpora5/LDC/LDC99S79 \
                           data/swbd2_phase2_train
local/make_swbd2_phase3.pl /export/corpora5/LDC/LDC2002S06 \
                           data/swbd2_phase3_train
local/make_swbd_cellular1.pl /export/corpora5/LDC/LDC2001S13 \
                             data/swbd_cellular1_train
local/make_swbd_cellular2.pl /export/corpora5/LDC/LDC2004S07 \
                             data/swbd_cellular2_train

# TODO create data prep script(s) for Callhome.

utils/combine_data.sh data/train \
  data/swbd_cellular1_train data/swbd_cellular2_train \
  data/swbd2_phase2_train data/swbd2_phase3_train data/sre

for name in sre train callhome; do
  steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 40 --cmd "$train_cmd" \
      data/$name exp/make_mfcc $mfccdir
  utils/fix_data_dir.sh data/$name
done

for name in sre train callhome; do
  sid/compute_vad_decision.sh --nj 40 --cmd "$train_cmd" \
      data/$name exp/make_vad $vaddir
  utils/fix_data_dir.sh data/$name
done

# Reduce the amount of training data for the UBM.
utils/subset_data_dir.sh data/train 16000 data/train_16k
utils/subset_data_dir.sh data/train 32000 data/train_32k

# Train UBM and i-vector extractor.
diarization/train_diag_ubm.sh --cmd "$train_cmd -l mem_free=20G,ram_free=20G" \
  --nj 20 --num-threads 8 \
  --delta-order 1 \
  data/train_16k $num_components \
  exp/diag_ubm_$num_components

diarization/train_full_ubm.sh --nj 40 --remove-low-count-gaussians false \
  --cmd "$train_cmd -l mem_free=25G,ram_free=25G" data/train_32k \
  exp/diag_ubm_$num_components exp/full_ubm_$num_components

diarization/train_ivector_extractor.sh --cmd "$train_cmd -l mem_free=35G,ram_free=35G" \
  --ivector-dim $ivector_dim \
  --num-iters 5 exp/full_ubm_$num_components/final.ubm data/train \
  exp/extractor_c${num_components}_i${ivector_dim}

diarization/extract_ivectors.sh --cmd "$train_cmd --mem 25G" \
  --nj 40 --use-vad true --chunk-size 300 --period 4500 \
  --min-chunk-size 100 exp/extractor_c${num_components}_i${ivector_dim} \
  data/sre exp/ivectors_sre

exit 1;
# The rest of this script is TODO, for now

