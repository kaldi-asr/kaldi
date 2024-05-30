#!/usr/bin/env bash

# Copyright 2016   Vimal Manohar
# Apache 2.0.

# See README.txt for more info on data required.

. ./cmd.sh
. ./path.sh

set -o pipefail
set -e

mfccdir=`pwd`/mfcc
nj=40
stage=-1

. utils/parse_options.sh

# Training corpora

# 1996 English Broadcast News Train (HUB4)
hub4_96_train_transcripts=/export/corpora/LDC/LDC97T22/hub4_eng_train_trans
hub4_96_train_speech=/export/corpora/LDC/LDC97S44/data
# 1997 English Broadcast News Train (HUB4)
hub4_97_train_transcripts=/export/corpora/LDC/LDC98T28/hub4e97_trans_980217
hub4_97_train_speech=/export/corpora/LDC/LDC98S71/97_eng_bns_hub4
# 1996 CSR HUB4 Language Model
csr_hub4_lm=/export/corpora/LDC/LDC98T31/1996_csr_hub4_model
# 1995 CSR-IV HUB4 corpus
csr95_hub4=/export/corpora/LDC/LDC96S31/csr95_hub4
# North American News Text Corpus
NA_text=/export/corpora/LDC/LDC95T21
# North American News Text Supplement Corpus
NA_text_supp=/export/corpora/LDC/LDC98T30/northam_news_txt_sup

# Test corpora

# 1996 English Broadcast News Dev and Eval (HUB4)
hub4_96_eval=/export/corpora/LDC/LDC97S66/1996_eng_bcast_dev_eval
# 1997 HUB4 English Evaluation corpus
hub4_97_eval=/export/corpora/LDC/LDC2002S11/hub4e_97
# 1998 HUB4 Broadcast News Evaluation English Test Material
hub4_98_eval=/export/corpora/LDC/LDC2000S86
# 1999 HUB4 Broadcast News Evaluation English Test Material
hub4_99_eval=/export/corpora5/LDC/LDC2000S88/hub4_1999

# Test sets used -- Uncomment and keep only test sets needed
test_sets="eval97.pem"
# test_sets="dev96ue dev96pe eval96 eval96.pem eval97 eval97.pem eval98 eval98.pem eval99_1 eval99_1.pem eval99_2 eval99_2.pem"

if [ $stage -le 0 ]; then
  # Prepare 1996 English Broadcast News Train (HUB4)
  local/data_prep/prepare_1996_bn_data.sh \
    $hub4_96_train_transcripts \
    $hub4_96_train_speech \
    data/local/data/train_bn96

  # Prepare 1997 English Broadcast News Train (HUB4)
  local/data_prep/prepare_1997_bn_data.sh \
    $hub4_97_train_transcripts \
    $hub4_97_train_speech \
    data/local/data/train_bn97
fi

# Install Beautiful Soup 4 python package for parsing SGML-like files
# in CSR-IV HUB4 corpus
if [ ! -d tools/beautifulsoup4 ]; then
  mkdir -p tools
  pip install -t tools/beautifulsoup4 beautifulsoup4
fi
export PYTHONPATH=$PWD/tools/beautifulsoup4:$PYTHONPATH

if [ $stage -le 1 ]; then
  if [ ! -f $csr_hub4_lm/utils.tar ]; then
    echo "Expected CSR-IV utils.tar to be found"
    exit 1
  fi

  mkdir -p tools/csr4_utils
  (
    cd tools/csr4_utils
    tar -xvf $csr_hub4_lm/utils.tar
  )

  chmod a+w tools/csr4_utils
  patch -u -d tools/csr4_utils -p3 < local/data_prep/csr4_utils.patch
fi

if [ $stage -le 2 ]; then
  # Prepare 1995 CSR-IV HUB4 corpus
  local/data_prep/prepare_1995_csr_hub4_corpus.sh \
    $csr95_hub4 data/local/data/csr95_hub4
fi

if [ $stage -le 3 ]; then
  # Prepare North American News Text Corpus
  local/data_prep/prepare_na_news_text_corpus.sh --nj 40 --cmd "$train_cmd" \
     $NA_text data/local/data/na_news

  # Prepare North American News Text Supplement Corpus
  local/data_prep/prepare_na_news_text_supplement.sh --nj 10 --cmd "$train_cmd" \
    $NA_text_supp data/local/data/na_news_supp
fi

if [ $stage -le 4 ]; then
  # Prepare 1996 CSR HUB4 Language Model
  local/data_prep/prepare_1996_csr_hub4_lm_corpus.sh --nj 10 --cmd "$train_cmd" \
     $csr_hub4_lm data/local/data/csr96_hub4
fi

if [ $stage -le 5 ]; then
  # Prepare 1996 English Broadcast News Dev and Eval (HUB4)
  local/data_prep/prepare_1996_hub4_bn_eng_dev_and_eval.sh \
    $hub4_96_eval \
    data/local/data/hub4_96_dev_eval

  # Prepare 1997 HUB4 English Evaluation corpus
  local/data_prep/prepare_1997_hub4_bn_eng_eval.sh \
    $hub4_97_eval data/local/data/eval97

  # Prepare 1998 HUB4 Broadcast News Evaluation English Test Material
  local/data_prep/prepare_1998_hub4_bn_eng_eval.sh \
    $hub4_98_eval data/local/data/eval98

  # Prepare 1999 HUB4 Broadcast News Evaluation English Test Material
  local/data_prep/prepare_1999_hub4_bn_eng_eval.sh \
    $hub4_99_eval data/local/data/eval99
fi

if [ $stage -le 6 ]; then
  local/format_data.sh
fi

if [ $stage -le 7 ]; then
  local/train_lm.sh
fi

if [ $stage -le 8 ]; then
  local/prepare_dict.sh --dict-suffix "_nosp" \
    data/local/local_lm/data/work/wordlist

  utils/prepare_lang.sh data/local/dict_nosp \
    "<unk>" data/local/lang_tmp_nosp data/lang_nosp
fi

if [ $stage -le 9 ]; then
  local/format_lms.sh --local-lm-dir data/local/local_lm
fi

if [ $stage -le 10 ]; then
  for x in train $test_sets; do
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
fi

if [ $stage -le 15 ]; then
  utils/subset_data_dir.sh --shortest data/train 1000 data/train_1kshort
  utils/subset_data_dir.sh data/train 2000 data/train_2k

  # Note: the --boost-silence option should probably be omitted by default
  # for normal setups.  It doesn't always help. [it's to discourage non-silence
  # models from modeling silence.]
  steps/train_mono.sh --boost-silence 1.25 --nj $nj --cmd "$train_cmd" \
    data/train_1kshort data/lang_nosp exp/mono0a
fi

if [ $stage -le 16 ]; then
  steps/align_si.sh --boost-silence 1.25 --nj $nj --cmd "$train_cmd" \
    data/train_2k data/lang_nosp exp/mono0a exp/mono0a_ali

  steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" 2000 10000 \
    data/train_2k data/lang_nosp exp/mono0a_ali exp/tri1
fi

if [ $stage -le 17 ]; then
  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
    data/train data/lang_nosp exp/tri1 exp/tri1_ali

  steps/train_lda_mllt.sh --cmd "$train_cmd" 2500 15000 \
    data/train data/lang_nosp exp/tri1_ali exp/tri2
fi

if [ $stage -le 18 ]; then
  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
    data/train data/lang_nosp exp/tri2 exp/tri2_ali

  steps/train_sat.sh --cmd "$train_cmd" 4200 40000 \
    data/train data/lang_nosp exp/tri2_ali exp/tri3
fi

if [ $stage -le 19 ]; then
  utils/mkgraph.sh data/lang_nosp_test exp/tri3 exp/tri3/graph_nosp

  for dset in $test_sets; do
    (
    this_nj=`cat data/$dset/spk2utt | wc -l`
    if [ $this_nj -gt 20 ]; then
      this_nj=20
    fi
    steps/decode_fmllr.sh --nj $this_nj --cmd "$decode_cmd" --num-threads 4 \
      exp/tri3/graph_nosp data/$dset exp/tri3/decode_nosp_${dset} || touch exp/tri3/.error
    steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
      data/lang_nosp_test data/lang_nosp_test_rescore \
      data/${dset} exp/tri3/decode_nosp_${dset} \
      exp/tri3/decode_nosp_${dset}_rescore || touch exp/tri3/.error
    ) &
  done
  wait

  if [ -f exp/tri3/.error ]; then
    echo "Decode failed in exp/tri3/decode*"
    exit 1
  fi
fi

if [ $stage -le 20 ]; then
  steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" \
    data/train data/lang_nosp exp/tri3 exp/tri3_ali

  steps/train_sat.sh --cmd "$train_cmd" 5000 100000 \
    data/train data/lang_nosp exp/tri3_ali exp/tri4
fi

if [ $stage -le 21 ]; then
  utils/mkgraph.sh data/lang_nosp_test exp/tri4 exp/tri4/graph_nosp

  for dset in $test_sets; do
    (
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
    ) &
  done
  wait

  if [ -f exp/tri4/.error ]; then
    echo "Decode failed in exp/tri4/decode*"
    exit 1
  fi
fi

wait

# %WER 18.0 | 728 32834 | 83.9 11.7 4.3 2.0 18.0 85.9 | exp/tri4/decode_nosp_eval97.pem_rescore/score_14_0.0/eval97.pem.ctm.filt.sys
# %WER 19.3 | 728 32834 | 82.9 12.6 4.6 2.2 19.3 86.8 | exp/tri4/decode_nosp_eval97.pem/score_13_0.0/eval97.pem.ctm.filt.sys

# The following demonstrates how to use out-of-domain WSJ models to segment long
# audio recordings of HUB4 with raw unaligned transcripts into short segments
# with aligned transcripts for training new ASR models.

# local/run_segmentation_wsj.sh
exit 0
