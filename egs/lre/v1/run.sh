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

#local/make_sre_2008_train.pl local/language_abbreviations.txt /export/corpora5/LDC/LDC2011S05 data
#utils/combine_data.sh data/sre08_train data/sre08_train_10sec_female data/sre08_train_10sec_male \
#    data/sre08_train_3conv_female data/sre08_train_3conv_male data/sre08_train_8conv_female \
#    data/sre08_train_8conv_male data/sre08_train_short2_male data/sre08_train_short2_female /export/a14/kumar/kaldi/language_id/egs/lre/v1/data/ldc96s*

local/make_sre_2008_train.pl local/language_abbreviations.txt /export/corpora5/LDC/LDC2011S05 data
local/make_ldc96s.pl 49 /export/corpora5/LDC/LDC96S49 data
local/make_ldc96s.pl 54 /export/corpora5/LDC/LDC96S54 data
local/make_ldc96s.pl 55 /export/corpora5/LDC/LDC96S55 data
local/make_ldc96s.pl 56 /export/corpora5/LDC/LDC96S56 data
local/make_ldc96s.pl 57 /export/corpora5/LDC/LDC96S57 data
local/make_ldc96s.pl 58 /export/corpora5/LDC/LDC96S58 data
utils/combine_data.sh data/all data/sre08_train_10sec_female data/sre08_train_10sec_male \
    data/sre08_train_3conv_female data/sre08_train_3conv_male data/sre08_train_8conv_female \
    data/sre08_train_8conv_male data/sre08_train_short2_male data/sre08_train_short2_female data/ldc96s*

mfccdir=`pwd`/mfcc
vaddir=`pwd`/mfcc

set -e
steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 40 --cmd "$train_cmd" data/all exp/make_mfcc $mfccdir

lid/compute_vad_decision.sh --nj 4 --cmd "$train_cmd" data/all exp/make_vad $vaddir

# Use 4k of the 14k utterances for testing, but make sure the speakers do not
# overlap with the rest of the data, which will be used for training.
utils/subset_data_dir.sh --speakers data/all 4000 data/test
utils/filter_scp.pl --exclude data/test/spk2utt < data/all/spk2utt  | awk '{print $1}' > foo
utils/subset_data_dir.sh --spk-list foo data/all data/train


utils/subset_data_dir.sh data/train 3000 data/train_3k
utils/subset_data_dir.sh data/train 6000 data/train_6k


lid/train_diag_ubm.sh --nj 30 --cmd "$train_cmd" data/train_3k 2048 exp/diag_ubm_2048
lid/train_full_ubm.sh --nj 30 --cmd "$train_cmd" data/train_6k exp/diag_ubm_2048 exp/full_ubm_2048_6k

lid/train_full_ubm.sh --nj 30 --cmd "$train_cmd" data/train exp/full_ubm_2048_6k exp/full_ubm_2048


lid/train_ivector_extractor.sh --cmd "$train_cmd -l mem_free=2G,ram_free=2G" \
  --num-iters 5 exp/full_ubm_2048/final.ubm data/train \
  exp/extractor_2048

lid/extract_ivectors.sh --cmd "$train_cmd -l mem_free=3G,ram_free=3G" --nj 50 \
   exp/extractor_2048 data/train exp/ivectors_train
