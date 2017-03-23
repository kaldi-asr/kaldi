#!/bin/bash

# Copyright 2016   Vimal Manohar
# Apache 2.0.

# See README.txt for more info on data required.

. cmd.sh
. path.sh

set -o pipefail

mfccdir=`pwd`/mfcc
nj=40

false && {
# Prepare 1996 English Broadcast News Train (HUB4)
local/data_prep/prepare_1996_bn_data.py --noise-word="<NOISE>" \
  --spoken-noise-word="<SPOKEN_NOISE>" \
  /export/corpora/LDC/LDC97S44 /export/corpora/LDC/LDC97T22 \
  data/local/data/train_bn96

# Prepare 1995 CSR-IV HUB4 corpus
local/data_prep/prepare_1995_csr_hub4_corpus.sh \
  /export/corpora5/LDC/LDC96S31/csr95_hub4/ data/local/data/csr95_hub4

# Prepare North American News Text Corpus
local/data_prep/prepare_na_news_text_corpus.sh --nj 40 --cmd "$train_cmd" \
  /export/corpora/LDC/LDC95T21 data/local/data/na_news

# Prepare North American News Text Supplement Corpus
local/data/prep/prepare_na_news_text_supplement.sh --nj 10 --cmd "$train_cmd" \
  /export/corpura/LCD/LDC98T30/northam_news_txt_sup data/local/data/na_news_supp

# Prepare 1996 CSR HUB4 Language Model
local/data_prep/prepare_1996_csr_hub4_lm_corpus.sh --nj 10 --cmd "$train_cmd" \
  /export/corpora/LDC/LDC98T31/1996_csr_hub4_model data/local/data/csr96_hub4

# Prepare 1996 English Broadcast News Dev and Eval (HUB4)
local/data_prep/prepare_1996_hub4_bn_eng_dev_and_eval.sh \
  /export/corpora/LDC/LDC97S66/1996_eng_bcast_dev_eval \
  data/local/data/hub4_96_dev_eval

# Prepare 1997 HUB4 English Evaluation corpus
local/data_prep/prepare_1997_hub4_bn_eng_eval.sh \
  /export/corpora/LDC/LDC2002S11/hub4e_97 data/local/data/eval97

# Prepare 1998 HUB4 Broadcast News Evaluation English Test Material
local/data_prep/prepare_1998_hub4_bn_eng_eval.sh \
  /export/corpora/LDC/LDC2000S86/ data/local/data/eval98

# Prepare 1999 HUB4 Broadcast News Evaluation English Test Material
local/data_prep/prepare_1999_hub4_bn_eng_eval.sh \
  /export/corpora5/LDC/LDC2000S88/hub4_1999 data/local/data/eval99

local/format_data.sh 

local/train_lm.sh 

local/prepare_dict.sh --dict-suffix "_nosp" \
  data/local/local_lm/data/work/wordlist

utils/prepare_lang.sh data/local/dict_nosp \
  "<unk>" data/local/lang_tmp_nosp data/lang_nosp

local/format_lms.sh --local-lm-dir data/local/local_lm

for x in train dev96ue dev96pe eval96 eval96.pem eval97 eval97.pem eval98 eval98.pem eval99_1 eval99_1.pem eval99_2 eval99_2.pem; do 
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

(
for dset in eval97.pem; do
  this_nj=`cat data/$dset/spk2utt | wc -l`
  if [ $this_nj -gt 20 ]; then
    this_nj=20
  fi
  steps/decode_fmllr.sh --nj $this_nj --cmd "$decode_cmd" --num-threads 4 \
    exp/tri3/graph_nosp data/$dset exp/tri3/decode_nosp_${dset}
  steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
    data/lang_nosp_test data/lang_nosp_test_rescore \
    data/${dset} exp/tri3/decode_nosp_${dset} \
    exp/tri3/decode_nosp_${dset}_rescore
done
) &

steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" \
  data/train data/lang_nosp exp/tri3 exp/tri3_ali

steps/train_sat.sh --cmd "$train_cmd" 5000 100000 \
  data/train data/lang_nosp exp/tri3_ali exp/tri4

utils/mkgraph.sh data/lang_nosp_test exp/tri4 exp/tri4/graph_nosp
}

for dset in eval97.pem; do
  this_nj=`cat data/$dset/spk2utt | wc -l`
  if [ $this_nj -gt 20 ]; then
    this_nj=20
  fi
  steps/decode_fmllr.sh --nj $this_nj --cmd "$decode_cmd" --num-threads 4 \
    exp/tri4/graph_nosp data/$dset exp/tri4/decode_nosp_${dset}
  steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
    data/lang_nosp_test data/lang_nosp_test_rescore \
    data/${dset} exp/tri4/decode_nosp_${dset} \
    exp/tri4/decode_nosp_${dset}_rescore
done

wait
exit 0
