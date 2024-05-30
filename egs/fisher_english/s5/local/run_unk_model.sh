#!/usr/bin/env bash

# Copyright 2017  Vimal Manohar

# This script prepares lang directory with UNK modeled by a phone LM.

utils/lang/make_unk_lm.sh data/local/dict exp/unk_lang_model || exit 1

utils/prepare_lang.sh \
  --unk-fst exp/unk_lang_model/unk_fst.txt \
  data/local/dict "<unk>" data/local/lang data/lang_unk

# note: it's important that the LM we built in data/lang/G.fst was created using
# pocolm with the option --limit-unk-history=true (see ted_train_lm.sh).  This
# keeps the graph compact after adding the unk model (we only have to add one
# copy of it).

exit 0

## Caution: if you use this unk-model stuff, be sure that the scoring script
## does not use lattice-align-words-lexicon, because it's not compatible with
## the unk-model.  Instead you should use lattice-align-words (of course, this
## only works if you have position-dependent phones).
