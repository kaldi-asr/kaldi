#!/usr/bin/env bash
# Copyright 2015-2016  Sarah Flora Juan
# Copyright 2016  Johns Hopkins University (Author: Yenda Trmal)
# Apache 2.0

set -e -o pipefail

# To create G.fst from ARPA language model
. ./path.sh || die "path.sh expected";

local/train_lms_srilm.sh --train-text data/train/text data/ data/srilm

nl -nrz -w10  corpus/LM/iban-bp-2012.txt | utils/shuffle_list.pl > data/local/external_text
local/train_lms_srilm.sh --train-text data/local/external_text data/ data/srilm_external

# let's do ngram interpolation of the previous two LMs
# the lm.gz is always symlink to the model with the best perplexity, so we use that

mkdir -p data/srilm_interp
for w in 0.9 0.8 0.7 0.6 0.5; do
    ngram -lm data/srilm/lm.gz  -mix-lm data/srilm_external/lm.gz \
          -lambda $w -write-lm data/srilm_interp/lm.${w}.gz
    echo -n "data/srilm_interp/lm.${w}.gz "
    ngram -lm data/srilm_interp/lm.${w}.gz -ppl data/srilm/dev.txt | paste -s -
done | sort  -k15,15g  > data/srilm_interp/perplexities.txt

# for basic decoding, let's use only a trigram LM
[ -d data/lang_test/ ] && rm -rf data/lang_test
cp -R data/lang data/lang_test
lm=$(cat data/srilm/perplexities.txt | grep 3gram | head -n1 | awk '{print $1}')
local/arpa2G.sh $lm data/lang_test data/lang_test

# for decoding using bigger LM let's find which interpolated gave the most improvement
[ -d data/lang_big ] && rm -rf data/lang_big
cp -R data/lang data/lang_big
lm=$(cat data/srilm_interp/perplexities.txt | head -n1 | awk '{print $1}')
local/arpa2G.sh $lm data/lang_big data/lang_big

# for really big lm, we should only decode using small LM
# and resocre using the big lm
utils/build_const_arpa_lm.sh $lm data/lang_big data/lang_big
exit 0;
