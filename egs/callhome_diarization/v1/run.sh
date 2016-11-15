#!/bin/bash
# Copyright 2016  David Snyder
# TODO other authors
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

# Prepare a collection of NIST SRE data. This will be used to train the UBM,
# iVector extractor and PLDA model.
local/make_sre.sh data

# Prepare SWB for UBM and iVector extractor training.
local/make_swbd2_phase2.pl /export/corpora5/LDC/LDC99S79 \
                           data/swbd2_phase2_train
local/make_swbd2_phase3.pl /export/corpora5/LDC/LDC2002S06 \
                           data/swbd2_phase3_train
local/make_swbd_cellular1.pl /export/corpora5/LDC/LDC2001S13 \
                             data/swbd_cellular1_train
local/make_swbd_cellular2.pl /export/corpora5/LDC/LDC2004S07 \
                             data/swbd_cellular2_train

# TODO: Currently missing dataprep script for Callhome
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

# TODO: Split callhome into two parts, callhome1 and callhome2.
# Each partition is treated as a held-out dataset, and used to
# compute various quantities needed to perform diarization on
# the other part.

# Reduce the amount of training data for the UBM.
utils/subset_data_dir.sh data/train 16000 data/train_16k
utils/subset_data_dir.sh data/train 32000 data/train_32k

# Train UBM and i-vector extractor.
diarization/train_diag_ubm.sh --cmd "$train_cmd -l mem_free=20G,ram_free=20G" \
  --nj 20 --num-threads 8 --delta-order 1 data/train_16k $num_components \
  exp/diag_ubm_$num_components

diarization/train_full_ubm.sh --nj 40 --remove-low-count-gaussians false \
  --cmd "$train_cmd -l mem_free=25G,ram_free=25G" data/train_32k \
  exp/diag_ubm_$num_components exp/full_ubm_$num_components

diarization/train_ivector_extractor.sh \
  --cmd "$train_cmd -l mem_free=35G,ram_free=35G" \
  --ivector-dim $ivector_dim --num-iters 5 \
  exp/full_ubm_$num_components/final.ubm data/train \
  exp/extractor_c${num_components}_i${ivector_dim}

# TODO Extract iVectors for the SRE, which is our PLDA training
# data.  A long period is used here so that we don't compute too
# many iVectors for each recording.
diarization/extract_ivectors.sh --cmd "$train_cmd --mem 25G" \
  --nj 40 --use-vad true --chunk-size 300 --period 4500 \
  --min-chunk-size 100 exp/extractor_c${num_components}_i${ivector_dim} \
  data/sre exp/ivectors_sre

# TODO Extract iVectors for the two partitions of callhome.
diarization/extract_ivectors.sh --cmd "$train_cmd --mem 20G" \
  --nj 40 --use-vad false --chunk-size 150 --period 75 \
  --min-chunk-size 50 exp/extractor_c${num_components}_i${ivector_dim} \
  data/callhome1 exp/ivectors_callhome1

diarization/extract_ivectors.sh --cmd "$train_cmd --mem 20G" \
  --nj 40 --use-vad false --chunk-size 150 --period 75 \
  --min-chunk-size 50 exp/extractor_c${num_components}_i${ivector_dim} \
  data/callhome2 exp/ivectors_callhome2

# TODO: Probably this should be in its own script and submitted as a job
ivector-compute-plda ark:exp/ivectors_sre/spk2utt "ark:ivector-subtract-global-mean scp:exp/ivectors_sre/ivector.scp ark:- | transform-vec exp/ivectors_callhome1/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" exp/ivectors_callhome1/plda 2> exp/ivectors_callhome1/log/plda.log

ivector-compute-plda ark:exp/ivectors_sre/spk2utt "ark:ivector-subtract-global-mean scp:exp/ivectors_sre/ivector.scp ark:- | transform-vec exp/ivectors_callhome2/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" exp/ivectors_callhome2/plda 2> exp/ivectors_callhome2/log/plda.log

# TODO: This performs unsupervised calibration.  Each partition is used
# as a held-out dataset to compute the stopping criteria used
# to cluster the other partition.
diarization/compute_plda_calibration.sh --cmd "$train_cmd --mem 4G" \
  --nj 20 exp/ivectors_callhome2

diarization/compute_plda_calibration.sh --cmd "$train_cmd --mem 4G" \
  --nj 20 exp/ivectors_callhome1

# TODO: Perform PLDA scoring on all pairs of segments for each recording.
diarization/score_plda.sh --cmd "$train_cmd --mem 4G" \
  --nj 20 exp/ivectors_callhome2 exp/ivectors_callhome1

diarization/score_plda.sh --cmd "$train_cmd --mem 4G" \
  --nj 20 exp/ivectors_callhome1 exp/ivectors_callhome2

# TODO: Cluster the PLDA scores using agglomerative hierarchical clustering.
# Note that the stopping threshold is computed on a different partition of
# Callhome than the one we're currently clustering.
# TODO: Need an option to specify number of speakers. This should also be evaluated here.
diarization/cluster.sh --cmd "$train_cmd --mem 4G" \
  --nj 20 --threshold `cat exp/ivectors_callhome2/threshold.txt` \
  exp/ivectors_callhome1

diarization/cluster.sh --cmd "$train_cmd --mem 4G" \
  --nj 20 --threshold `cat exp/ivectors_callhome1/threshold.txt` \
  exp/ivectors_callhome2

python diarization/make_rttm.py exp/ivectors_callhome1/segments exp/ivectors_callhome1/labels.txt > callhome1.rttm
python diarization/make_rttm.py exp/ivectors_callhome2/segments exp/ivectors_callhome2/labels.txt > callhome2.rttm
cat callhome1.rttm callhome2.rttm > callhome.rttm

# OVERALL SPEAKER DIARIZATION ERROR = 10.32 percent of scored speaker time  `(ALL)
perl local/md-eval.pl -1 -c 0.25 -r local/fullref.rttm -s callhome.rttm | tee results.txt

