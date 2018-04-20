#!/bin/bash
# Copyright 2018  Xuechen Liu

# compare wer between diff. models in hkust nnet3 directory
# exemplar usage: local/nnet3/compare_wer.sh exp/nnet3/tdnn_sp exp/nnet3/tdnn_sp_pr43
# note: this script is made quite general since we kinda wanna give more flexibility to
#       users on adding affix for their own use when training models.

set -e
. ./cmd.sh
. ./path.sh

if [ $# == 0 ]; then
  echo "Usage: $0: [--online] <dir1> <dir2>"
  echo "e.g.: $0 exp/chain/tdnn_{b,c}_sp"
  echo "or (with epoch numbers for discriminative training):"
  echo "$0 exp/chain/tdnn_b_sp_disc:{1,2,3}"
  exit 1
fi

echo "# $0 $*"

modeldir_1=$0
modeldir_2=$1

# check required directories
for model in modeldir_1 modeldir_2; do
  [ -d "${model}/decode/" ] || (echo "model $model has no results for compare" && exit 1;)
  [ -d ${model}/ ]
done 2>/dev/null || exit 1;

# grep WER
wer_1=$(for x in $modeldir_1/decode; do [ -d $x ] && grep WER $x/cer_* | utils/best_wer.sh; done 2>/dev/null | awk '{print $2}')
wer_2=$(for x in $modeldir_2/decode; do [ -d $x ] && grep WER $x/cer_* | utils/best_wer.sh; done 2>/dev/null | awk '{print $2}')

# grep log prob for train and valid set
logprob_train_1=$(grep Overall ${modeldir_1}/log/compute_prob_train.{final,combined}.log 2>/dev/null | grep log-like | awk '{printf("%.4f", $8)}')
logprob_train_2=$(grep Overall ${modeldir_2}/log/compute_prob_train.{final,combined}.log 2>/dev/null | grep log-like | awk '{printf("%.4f", $8)}')
logprob_valid_1=$(grep Overall ${modeldir_1}/log/compute_prob_valid.{final,combined}.log 2>/dev/null | grep log-like | awk '{printf("%.4f", $8)}')
logprob_valid_2=$(grep Overall ${modeldir_2}/log/compute_prob_valid.{final,combined}.log 2>/dev/null | grep log-like | awk '{printf("%.4f", $8)}')

# form the table



