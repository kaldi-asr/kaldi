#!/usr/bin/env bash

# Copyright 2019 Johns Hopkins University (author: Jinyi Yang)
# Apache 2.0

# This script interpolates two language models.

ngram_order=4
oov_sym="<UNK>"
prune_thres=1e-9

[ -f ./path.sh ] && . ./path.sh
. parse_options.sh || exit 1;

if [ $# != 4 ]; then
  echo "Usage: [--ngram-order] [--oov-sym] [--prune-thres] <lm-dir-1> <lm-dir-2> <lm-mix-dir> <dev-dir>"
  echo "E.g. $0 --ngram-order 4 --oov-sym \"<UNK>\" --prune-thres \"1e-9\" \
  data/local/lm_gale data/local/lm_giga data/local/lm_mix data/dev "
  exit 1;
fi
lm_dir_1=$1
lm_dir_2=$2
lm_mix_dir=$3
heldout=$4/text

mkdir -p $lm_mix_dir || exit 1;
if [ ! -f $lm_dir_mix/srilm.o${ngram_order}g.kn.gz ]; then
  for d in $lm_dir_1 $lm_dir_2; do
    ngram -debug 2 -order $ngram_order -unk -lm $d/srilm.o${ngram_order}g.kn.gz \
      -ppl $heldout > $d/ppl ;
  done
  compute-best-mix $lm_dir_1/ppl $lm_dir_2/ppl > $lm_mix_dir/best-mix.ppl
  lambdas=$(grep -o '(.*)' $lm_mix_dir/best-mix.ppl | head -1)
  lambdas=${lambdas%%)}
  lambdas=${lambdas##(}
  lambda1=`echo $lambdas | cut -d " " -f1`
  lambda2=`echo $lambdas | cut -d " " -f2`
  ngram_opts="$lm_dir_1/srilm.o${ngram_order}g.kn.gz -weight $lambda1 -order \
    $ngram_order\n$lm_dir_2/srilm.o${ngram_order}g.kn.gz -weight $lambda2 -order $ngram_order"
  echo -e ${ngram_opts} > $lm_mix_dir/ngram_opts
  ngram -order $ngram_order \
    -unk -map-unk $oov_sym \
    -prune $prune_thres \
    -read-mix-lms -lm $lm_mix_dir/ngram_opts \
    -write-lm $lm_mix_dir/srilm.o${ngram_order}g.kn.gz
  ngram -debug 2 -order $ngram_order -unk \
  -lm $lm_mix_dir/srilm.o${ngram_order}g.kn.gz \
  -ppl $heldout > $lm_mix_dir/lm.ppl
fi
echo "LM interpolation done"
