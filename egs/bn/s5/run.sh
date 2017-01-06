#!/bin/bash

# Copyright 2016   Vimal Manohar
# Apache 2.0.

# See README.txt for more info on data required.

. cmd.sh
. path.sh

set -o pipefail

mfccdir=`pwd`/mfcc
nj=40

local/data_prep/prepare_bn_data.py --split-at-sync=false \
  /export/corpora5/LDC/LDC97S44 \
  /export/corpora/LDC/LDC97T22 data/local/data/train

local/data_prep/prepare_na_news_test_corpus.sh --nj 40 --cmd "$train_cmd" \
  /export/corpora/LDC/LDC95T21 data/local/data/na_news

local/data_prep/prepare_1996_csr_hub4_corpus.sh --nj 10 --cmd "$train_cmd" \
  /export/corpora/LDC/LDC98T31 data/local/data/csr96_hub4

local/prepare_1998_hub4_bn_eng_eval.sh /export/corpora/LDC/LDC2000S86/ \
  data/local/data/eval98

local/format_data.sh 

local/train_lm.sh 

local/prepare_dict.sh --dict-suffix "_nosp" \
  data/local/local_lm/data/work/wordlist

utils/prepare_lang.sh data/local/dict_nosp \
  "<unk>" data/local/lang_tmp_nosp data/lang_nosp

local/format_lms.sh 

for x in train eval98 eval98.pem; do 
  this_nj=$(cat data/$x/utt2spk | wc -l)
  if [ $this_nj -gt 30 ]; then
    this_nj=30
  fi

  steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj $this_nj \
    --cmd "$train_cmd" \
    data/$x exp/make_mfcc $mfccdir
  steps/compute_cmvn_stats.sh data/$x exp/make_mfcc $mfccdir
  utils/fix_data_dir.sh data/$x
done

utils/subset_data_dir.sh --shortest data/train 1000 data/train_1kshort
utils/subset_data_dir.sh data/train 2000 data/train_2k

# Note: the --boost-silence option should probably be omitted by default
# for normal setups.  It doesn't always help. [it's to discourage non-silence
# models from modeling silence.]
steps/train_mono.sh --boost-silence 1.25 --nj $nj --cmd "$train_cmd" \
  data/train_1kshort data/lang_nosp exp/mono0a

steps/align_si.sh --boost-silence 1.25 --nj $nj --cmd "$train_cmd" \
  data/train_2k data/lang_nosp exp/mono0a exp/mono0a_ali

steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" 2000 10000 \
  data/train_2k data/lang_nosp exp/mono0a_ali exp/tri1

steps/align_si.sh --nj $nj --cmd "$train_cmd" \
  data/train data/lang_nosp exp/tri1 exp/tri1_ali

steps/train_lda_mllt.sh --cmd "$train_cmd" 2500 15000 \
  data/train data/lang_nosp exp/tri1_ali exp/tri2

steps/align_si.sh --nj $nj --cmd "$train_cmd" \
  data/train data/lang_nosp exp/tri2 exp/tri2_ali

steps/train_sat.sh --cmd "$train_cmd" 4200 40000 \
  data/train data/lang_nosp exp/tri2_ali exp/tri3

utils/mkgraph.sh data/lang_nosp_test exp/tri3 exp/tri3/graph_nosp

steps/decode_fmllr.sh --nj 20 --cmd "$decode_cmd" \
  exp/tri3/graph_nosp data/eval98.pem exp/tri3/decode_nosp_eval98.pem
steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
  data/lang_nosp_test data/lang_nosp_test_rescore \
  data/eval98.pem exp/tri3/decode_nosp_eval98.pem \
  exp/tri3/decode_rescore_nosp_eval98.pem

exit 0
