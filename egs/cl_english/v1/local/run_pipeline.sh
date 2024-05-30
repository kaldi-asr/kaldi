#!/usr/bin/env bash

# Copyright 2021  Behavox (Author: Hossein Hadian).
# Apache 2.0.

# Run an ASR pipeline from the GMM training stage up to the chain model training.
# Expects the data/lang to be prepared already. Specifially, these should
# exist before running this script:
# -- data/$train_set
# -- data/lang_nosp with <unk> as OOV symbol
# -- data/local/dict_nosp should exist and
# -- $lm
# Where $lm and $train_set are given as input args.

. ./cmd.sh
. ./path.sh


stage=0
gmm_stage=0
nj=30
size=small   # small (30-100 hrs), medium (100-500 hrs), big (500+ hrs)
dim_opts=
dir=
exp=
softmax=false  # LWF needs the output layer to include softmax
affix=
tree_opts=
leaves=5000
tree_affix=
lm=data/local/lm.gz

echo "$0 $@"  # Print the command line for logging

. utils/parse_options.sh

if [ $# != 1 ]; then
  echo "Usage: $0 <train_set>"
  echo "E.g.: $0 fsh_train_200hr "
  exit 1;
fi

train_set=$1

set -euo pipefail

RED='\033[0;31m'
NC='\033[0m'

mkdir -p data/local

if [ -z $dim_opts ]; then
  if [ "$size" == "small" ]; then
    dim_opts="--dim 1024 --bdim 128"
  elif [ "$size" == "medium" ]; then
    dim_opts="--dim 1408 --bdim 160"
  fi
fi

[ -z $exp ] && exp=exp_${train_set}
[ -z $dir ] && dir=$exp/chain/tdnn1a_noiv_${size}_sp${affix}


echo "$0: train set is $train_set and exp is $exp....."
echo "$0: dim opts is $dim_opts"
sleep 3

if [ $stage -le 0 ]; then
  echo -e "$RED $0: Stage 0: Checking files...$NC"
  # Make sure all data exists as needed for CL.

  for f in data/$train_set/text $lm data/lang_nosp data/local/dict_nosp; do
    [ ! -e $f ] && echo "$0: no such file/dir $f" && exit 1;
  done

fi

if [ $stage -le 1 ]; then
  echo -e "$RED $0: Stage 1: GMM training.$NC"
  if [[ -f $exp/tri3b/final.mdl ]] && [[ -e $exp/dict ]]; then
    echo "$0: Final gmm model already exists."
  else
    local/run_gmm_common.sh \
      --stage $gmm_stage --exp $exp \
      --lm "$lm" --nj $nj --train $train_set --test-sets "" \
      --num-mono-utts 10000 --deltas-leaves "1500 10000" --lda-leaves "2500 20000" \
      --sat-leaves "4000 45000" --tri3-lda true --num-deltas-utts 30000 \
      --num-lda-utts 60000
  fi
fi

if [ $stage -le 2 ]; then
  echo -e "$RED $0: Stage 2: Chain no-ivector training.$NC"
  if [ -f $dir/final.mdl ]; then
    echo "$0: $dir/final.mdl already exists."
  else
    local/chain/train_tdnn_noivector_1a.sh \
      --softmax $softmax \
      --stage -10 --train-stage -10 --epochs 6 \
      --skip_decoding true \
      $dim_opts \
      --tree-opts "$tree_opts" --leaves $leaves --tree-affix "$tree_affix" \
      --exp $exp --dir $dir \
      --nj $nj \
      --train-set "$train_set" \
      --test-sets "fsh_dev"
  fi
fi

if [ $stage -le 3 ]; then
  echo -e "$RED $0: Stage 3: Decoding.$NC"
  local/run_evaluation.sh --lm $lm --stage 0 --iter final $dir
fi

echo Done
