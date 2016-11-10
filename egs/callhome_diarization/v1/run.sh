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
sid/train_diag_ubm.sh --cmd "$train_cmd -l mem_free=20G,ram_free=20G" \
  --nj 20 --num-threads 8 \
  --delta-order 1 \
  data/train_16k $num_components \
  exp/diag_ubm_$num_components

sid/train_full_ubm.sh --nj 40 --remove-low-count-gaussians false \
  --cmd "$train_cmd -l mem_free=25G,ram_free=25G" data/train_32k \
  exp/diag_ubm_$num_components exp/full_ubm_$num_components

sid/train_ivector_extractor.sh --cmd "$train_cmd -l mem_free=35G,ram_free=35G" \
  --ivector-dim $ivector_dim \
  --num-iters 5 exp/full_ubm_$num_components/final.ubm data/train \
  exp/extractor_c${num_components}_i${ivector_dim}

exit 1;
# The rest of this script is TODO, for now

sid/extract_ivectors_dense.sh --cmd "$train_cmd --mem 15G" \
  --chunk-size 128 --period 64 \
  exp/extractor_c${num_components}_i${ivector_dim} data/callhome_1000 \
  exp/ivectors_callhome_1000

utils/subset_data_dir.sh data/sre 8000 data/sre_8k

awk '{print $1, $1}' data/sre_8k/utt2spk > data/sre_8k/utt2spk.bak
awk '{print $1, $1}' data/sre_8k/utt2spk > data/sre_8k/spk2utt
mv data/sre_8k/utt2spk.bak data/sre_8k/utt2spk
rm data/sre_8k/spk2gender

utils/fix_data_dir.sh data/sre_8k

sid/resegement_from_vad.sh --cmd "$train_cmd" --max-utt 128000 \
  data/sre_8k data/sre_8k_seg exp/sre_8k_seg

sid/extract_ivectors.sh --cmd "$train_cmd --mem 25G" \
  --use-vad false \
  exp/extractor_c${num_components}_i${ivector_dim} data/sre_8k_seg \
  exp/ivectors_sre_8k

# TODO split callhome into two parts.
sid/extract_ivectors_dense.sh --cmd "$train_cmd --mem 15G" \
  --chunk-size 128 --period 64 \
  exp/extractor_c${num_components}_i${ivector_dim} data/callhome1 \
  exp/ivectors_callhome1

sid/extract_ivectors_dense.sh --cmd "$train_cmd --mem 15G" \
  --chunk-size 128 --period 64 \
  exp/extractor_c${num_components}_i${ivector_dim} data/callhome2 \
  exp/ivectors_callhome2

est-pca --read-vectors=true --normalize-mean=false --normalize-variance=true --dim=128 scp:exp/ivectors_callhome1/ivector.scp \
  exp/ivectors_callhome1/white.mat

est-pca --read-vectors=true --normalize-mean=false --normalize-variance=true --dim=128 scp:exp/ivectors_callhome2/ivector.scp \
  exp/ivectors_callhome2/white.mat

ivector-mean scp:exp/ivectors_callhome1/ivector.scp exp/ivectors_callhome1/mean.vec
ivector-mean scp:exp/ivectors_callhome2/ivector.scp exp/ivectors_callhome2/mean.vec

ivector-compute-plda --num-em-iters=20 \
   ark:data/sre_8k_seg/spk2utt "ark:ivector-subtract-global-mean scp:exp/ivectors_sre_8k/ivector.scp ark:- | transform-vec exp/ivectors_callhome1/white.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" exp/ivectors_sre_8k/plda1 2> exp/ivectors_sre_8k/log/plda1.log

ivector-compute-plda --num-em-iters=20 \
  ark:data/sre_8k_seg/spk2utt "ark:ivector-subtract-global-mean scp:exp/ivectors_sre_8k/ivector.scp ark:- | transform-vec exp/ivectors_callhome2/white.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" exp/ivectors_sre_8k/plda2 2> exp/ivectors_sre_8k/log/plda2.log

# Compute scores for callhome1 using models computed from callhome2
# TODO, scores don't need to be in text format
ivector-diarization-plda-scoring --target-energy=0.2 exp/ivectors_sre_8k/plda2 ark:exp/ivectors_callhome1/spk2utt \
   "ark:ivector-subtract-global-mean exp/ivectors_callhome2/mean.vec scp:exp/ivectors_callhome1/ivector.scp ark:- | transform-vec exp/ivectors_callhome2/white.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" ark,t:exp/ivectors_callhome1/scores.ark

# Compute scores for callhome2 using models computed from callhome1
# TODO, scores don't need to be in text format
ivector-diarization-plda-scoring --target-energy=0.2 exp/ivectors_sre_8k/plda1 ark:exp/ivectors_callhome2/spk2utt \
   "ark:ivector-subtract-global-mean exp/ivectors_callhome1/mean.vec scp:exp/ivectors_callhome2/ivector.scp ark:- | transform-vec exp/ivectors_callhome2/white.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" ark,t:exp/ivectors_callhome2/scores.ark
agglomerative-cluster --threshold=0.59 ark:exp/ivectors_callhome1/scores.ark ark:exp/ivectors_callhome1/spk2utt ark,t:exp/ivectors_callhome1/labels.txt

agglomerative-cluster --threshold=0.59 ark:exp/ivectors_callhome2/scores.ark ark:exp/ivectors_callhome2/spk2utt ark,t:exp/ivectors_callhome2/labels.txt

python make_rttm.py exp/ivectors_callhome1/segments exp/ivectors_callhome1/labels.txt > callhome1.rttm
python make_rttm.py exp/ivectors_callhome2/segments exp/ivectors_callhome2/labels.txt > callhome2.rttm
cat callhome1.rttm callhome2.rttm > callhome.rttm
perl local/md-eval.pl -1 -c 0.25 -r local/fullref.rttm -s callhome.rttm  >  out_bottom_up.txt
cat out_bottom_up.txt

