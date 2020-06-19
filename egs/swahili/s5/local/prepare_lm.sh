#!/usr/bin/env bash

. ./path.sh || die "path.sh expected";

cd data
#convert to FST format for Kaldi
arpa2fst --disambig-symbol=#0 --read-symbol-table=lang/words.txt \
  local/swahili.arpa lang/G.fst
