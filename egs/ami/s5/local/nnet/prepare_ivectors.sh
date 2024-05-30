#!/usr/bin/env bash

# Copyright 2016, Brno University of Technology (Author: Karel Vesely)
# Apache 2.0

. ./path.sh
. ./cmd.sh

train=data_ihm-fmllr-tri4/ihm/train
dev=data_ihm-fmllr-tri4/ihm/dev
eval=data_ihm-fmllr-tri4/ihm/eval
gmm=exp/ihm/tri4a

# Output directory for models and i-vectors,
ivec_absdir=$(readlink -m data_ihm-fmllr-tri4/ihm/ivector)

. utils/parse_options.sh

set -euxo pipefail

# UBM training (1024 components),
ubm=$ivec_absdir/ubm
steps/nnet/ivector/train_diag_ubm.sh --cmd "$train_cmd" --nj 10 \
  $train 1024 $ubm

# Training i-vector extractor (100 dims),
iextractor=$ivec_absdir/iextractor
steps/nnet/ivector/train_ivector_extractor.sh --cmd "$train_cmd --mem 5G" --nj 10 \
  --ivector-dim 100 $train $ubm $iextractor

# Extracting the i-vectors (per speaker, as the per-utterance copies),
steps/nnet/ivector/extract_ivectors.sh --cmd "$train_cmd" --nj 80 \
  $train data/lang $iextractor \
  ${gmm}_ali $ivec_absdir/ivec_train
steps/nnet/ivector/extract_ivectors.sh --cmd "$train_cmd" --nj 20 \
  $dev data/lang $iextractor \
  $gmm/decode_dev_ami_fsh.o3g.kn.pr1-7 $ivec_absdir/ivec_dev
steps/nnet/ivector/extract_ivectors.sh --cmd "$train_cmd" --nj 20 \
  $eval data/lang $iextractor \
  $gmm/decode_eval_ami_fsh.o3g.kn.pr1-7 $ivec_absdir/ivec_eval


# POST-PROCESS PER-SPEAKER I-VECTORS:

# Get the global mean of the i-vectors (train),
ivector-mean scp:$ivec_absdir/ivec_train/ivectors_spk.scp $iextractor/global_mean

# Merge the sets, normalize means, apply length normalization,
cat $ivec_absdir/ivec_train/ivectors_spk-as-utt.scp \
    $ivec_absdir/ivec_dev/ivectors_spk-as-utt.scp \
    $ivec_absdir/ivec_eval/ivectors_spk-as-utt.scp | \
  ivector-subtract-global-mean $iextractor/global_mean scp:- ark:- | \
  ivector-normalize-length --scaleup=false ark:- ark,scp:$ivec_absdir/ivectors_spk-as-utt_normalized.ark,$ivec_absdir/ivectors_spk-as-utt_normalized.scp


# POST-PROCESS PER-SENTENCE I-VECTORS:

# Get the global mean of the i-vectors (train, per-sentence),
ivector-mean scp:$ivec_absdir/ivec_train/ivectors_utt.scp $iextractor/global_mean_utt

# Merge the sets, normalize means, apply length normalization,
cat $ivec_absdir/ivec_train/ivectors_utt.scp \
    $ivec_absdir/ivec_dev/ivectors_utt.scp \
    $ivec_absdir/ivec_eval/ivectors_utt.scp | \
  ivector-subtract-global-mean $iextractor/global_mean_utt scp:- ark:- | \
  ivector-normalize-length --scaleup=false ark:- ark,scp:$ivec_absdir/ivectors_utt_normalized.ark,$ivec_absdir/ivectors_utt_normalized.scp


exit 0 # Done!
