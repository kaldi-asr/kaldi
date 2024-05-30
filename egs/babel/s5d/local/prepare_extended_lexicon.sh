#!/usr/bin/env bash
# Copyright (c) 2016, Johns Hopkins University ( Yenda Trmal <jtrmal@gmail.com> )
# License: Apache 2.0

# Begin configuration section.
unk_fraction_boost=1.0
num_sent_gen=12000000
num_prons=1000000
# End configuration section
set -e -o pipefail
set -o nounset                              # Treat unset variables as an error

# Extend the original lexicon.
# Will creates the files data/local/extend/{lexiconp.txt,oov2prob}.
local/extend_lexicon.sh --cmd "$train_cmd" --cleanup false \
  --num-sent-gen $num_sent_gen --num-prons $num_prons \
  data/local/lexicon.txt data/local/lang_ext data/dev2h/text


extend_lexicon_param=()
[ -f data/local/extend/original_oov_rates ] || exit 1;
unk_fraction=`cat data/local/extend/original_oov_rates |\
  grep "token" | awk -v x=$unk_fraction_boost '{print $NF/100.0*x}'`
extend_lexicon_param=(--cleanup false --unk-fraction $unk_fraction \
  --oov-prob-file data/local/lang_ext/oov2prob)

cp -r data/lang data/lang_ext
local/arpa2G.sh ${extend_lexicon_param[@]} \
  data/srilm/lm.gz data/lang_ext data/lang_ext

