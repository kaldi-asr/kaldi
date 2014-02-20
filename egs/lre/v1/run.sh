#!/bin/bash
# Copyright  2014   David Snyder
# Apache 2.0.
#
# An incomplete run.sh for this example. Currently this only trains up up a gender 
# independent UBM and ivector with the SRE08 training data.

. cmd.sh
. path.sh
set -e
mfccdir=`pwd`/mfcc
vaddir=`pwd`/mfcc

local/make_sre_2008_train.pl local/language_abbreviations.txt /export/corpora5/LDC/LDC2011S05 data
utils/combine_data.sh data/sre08_train_short2 data/sre08_train_short2_male data/sre08_train_short2_female

mfccdir=`pwd`/mfcc
vaddir=`pwd`/mfcc

set -e
steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 40 --cmd "$train_cmd" data/sre08_train_short2 exp/make_mfcc $mfccdir

lid/compute_vad_decision.sh --nj 4 --cmd "$train_cmd" data/sre08_train_short2 exp/make_vad $vaddir


lid/train_diag_ubm.sh --nj 30 --cmd "$train_cmd" data/sre08_train_short2 2048 exp/diag_ubm_2048
lid/train_full_ubm.sh --nj 30 --cmd "$train_cmd" data/sre08_train_short2 exp/diag_ubm_2048 exp/full_ubm_2048


lid/train_ivector_extractor.sh --cmd "$train_cmd -l mem_free=2G,ram_free=2G" \
  --num-iters 5 exp/full_ubm_2048/final.ubm data/sre08_train_short2 \
  exp/extractor_2048

lid/extract_ivectors.sh --cmd "$train_cmd -l mem_free=3G,ram_free=3G" --nj 50 \
   exp/extractor_2048 data/sre08_train_short2 exp/ivectors_sre08_train_short2
