#!/usr/bin/env bash
# Copyright 2019  QCRI (Author: Ahmed Ali)
# Apache 2.0

set -e -o pipefail

# To create G.fst from ARPA language model
. ./path.sh || die "path.sh expected";

local/train_lms_srilm.sh --train-text data/train/text data/ data/srilm

# for basic decoding, let's use only a trigram LM
[ -d data/lang_test/ ] && rm -rf data/lang_test
cp -R data/lang data/lang_test
lm=data/srilm/3gram.me.gz
utils/format_lm.sh data/lang_test $lm data/local/dict/lexicon.txt data/lang_test

# for decoding using bigger, we build 4-gram using the same transcription text
[ -d data/lang_big ] && rm -rf data/lang_big
cp -R data/lang data/lang_big
lm=data/srilm/4gram.me.gz
utils/format_lm.sh data/lang_big $lm data/local/dict/lexicon.txt data/lang_big

utils/build_const_arpa_lm.sh $lm data/lang_big data/lang_big
exit 0;
