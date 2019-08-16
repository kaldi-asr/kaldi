#!/bin/bash

# Copyright 2019 Microsoft Corp. (Authors: Xingyu Na)
# Apache 2.0

. ./cmd.sh
. ./path.sh

stage=0
dbase=/mnt/data/openslr
aidatatang_url=www.openslr.org/resources/62
aishell_url=www.openslr.org/resources/33
magicdata_url=www.openslr.org/resources/68
primewords_url=www.openslr.org/resources/47
stcmds_url=www.openslr.org/resources/38
thchs_url=www.openslr.org/resources/18

. utils/parse_options.sh

if [ $stage -le 0 ]; then
  # download all training data
  local/aidatatang_download_and_untar.sh $dbase/aidatatang $aidatatang_url aidatatang_200zh || exit 1; 
  local/aishell_download_and_untar.sh $dbase/aishell $aishell_url data_aishell || exit 1;
  local/magicdata_download_and_untar.sh $dbase/magicdata $magicdata_url train_set || exit 1;
  local/primewords_download_and_untar.sh $dbase/primewords $primewords_url || exit 1;
  local/stcmds_download_and_untar.sh $dbase/stcmds $stcmds_url || exit 1;
  local/thchs_download_and_untar.sh $dbase/thchs $thchs_url data_thchs30 || exit 1;

  # download all test data
  local/thchs_download_and_untar.sh $dbase/thchs $thchs_url test-noise || exit 1;
  local/magicdata_download_and_untar.sh $dbase/magicdata $magicdata_url dev_set || exit 1;
  local/magicdata_download_and_untar.sh $dbase/magicdata $magicdata_url test_set || exit 1;
fi

if [ $stage -le 1 ]; then
  local/aidatatang_data_prep.sh $dbase/aidatatang/aidatatang_200zh data/aidatatang || exit 1;
  local/aishell_data_prep.sh $dbase/aishell/data_aishell data/aishell || exit 1;
  local/thchs-30_data_prep.sh $dbase/thchs/data_thchs30 data/thchs || exit 1;
  local/magicdata_data_prep.sh $dbase/magicdata data/magicdata || exit 1;
  local/primewords_data_prep.sh $dbase/primewords data/primewords || exit 1;
  local/stcmds_data_prep.sh $dbase/stcmds data/stcmds || exit 1;
fi

if [ $stage -le 2 ]; then
  # normalize transcripts
  utils/combine_data.sh data/train_combined \
    data/{aidatatang,aishell,magicdata,primewords,stcmds,thchs}/train || exit 1;
  utils/combine_data.sh data/test_combined \
    data/{aidatatang,aishell,magicdata,thchs}/{dev,test} || exit 1;
  local/prepare_dict.sh
fi

if [ $stage -le 3 ]; then
  # train LM using transcription
  local/train_lms.sh
fi

if [ $stage -le 4 ]; then
  # prepare LM
  utils/prepare_lang.sh data/local/dict "<UNK>" data/local/lang data/lang
  utils/format_lm.sh data/lang data/local/lm/3gram-mincount/lm_unpruned.gz \
    data/local/dict/lexicon.txt data/lang_combined_tg
  utils/format_lm.sh data/lang data/local/lm/4gram-mincount/lm_unpruned.gz \
    data/local/dict/lexicon.txt data/lang_combined_fg
fi

if [ $stage -le 5 ]; then
  # make features
  mfccdir=mfcc
  corpora="aidatatang aishell magicdata primewords stcmds thchs"
  for c in $corpora; do
    (
      steps/make_mfcc_pitch_online.sh --cmd "$train_cmd" --nj 20 \
        data/$c/train exp/make_mfcc/$c/train || exit 1;
      steps/compute_cmvn_stats.sh data/$c/train \
        exp/make_mfcc/$c/train || exit 1;
    ) &
  done
  wait
fi

if [ $stage -le 6 ]; then
  # train mono and tri1a using aishell(~120k)
  # mono has been used in aishell recipe, so no test
  steps/train_mono.sh --boost-silence 1.25 --nj 20 --cmd "$train_cmd" \
    data/aishell/train data/lang exp/mono || exit 1;

  steps/align_si.sh --boost-silence 1.25 --nj 20 --cmd "$train_cmd" \
    data/aishell/train data/lang exp/mono exp/mono_ali || exit 1;
  steps/train_deltas.sh --boost-silence 1.25 --nj 20 --cmd "$train_cmd" 2500 20000 \
    data/aishell/train data/lang exp/mono_ali exp/tri1a || exit 1;
fi

if [ $stage -le 7 ]; then
  # train tri1b using aishell + primewords + stcmds + thchs (~280k)
  utils/combine_data.sh data/train_280k \
    data/{aishell,primewords,stcmds,thchs}/train || exit 1;

  steps/align_si.sh --boost-silence 1.25 --nj 40 --cmd "$train_cmd" \
    data/train_280k data/lang exp/tri1a exp/tri1a_280k_ali || exit 1;
  steps/train_deltas.sh --boost-silence 1.25 --nj 40 --cmd "$train_cmd" 4500 36000 \
    data/train_280k data/lang exp/tri1a_280k_ali exp/tri1b || exit 1;
fi

if [ $stage -le 8 ]; then
  # train tri2a using train_280k
  steps/align_si.sh --boost-silence 1.25 --nj 40 --cmd "$train_cmd" \
    data/train_280k data/lang exp/tri1b exp/tri1b_280k_ali || exit 1;
  steps/train_deltas.sh --boost-silence 1.25 --nj 40 --cmd "$train_cmd" 5500 90000 \
    data/train_280k data/lang exp/tri1b_280k_ali exp/tri2a || exit 1;
fi

if [ $stage -le 9 ]; then
  # train tri3a using aidatatang + aishell + primewords + stcmds + thchs (~440k)
  utils/combine_data.sh data/train_440k \
    data/{aidatatang,aishell,primewords,stcmds,thchs}/train || exit 1;

  steps/align_si.sh --boost-silence 1.25 --nj 60 --cmd "$train_cmd" \
    data/train_440k data/lang exp/tri2a exp/tri2a_440k_ali || exit 1;
  steps/train_lda_mllt.sh --cmd "$train_cmd" 7000 110000 \
    data/train_440k data/lang exp/tri2a_440k_ali exp/tri3a || exit 1;
fi

if [ $stage -le 10 ]; then
  # train tri4a using all
  utils/combine_data.sh data/train_all \
    data/{aidatatang,aishell,magicdata,primewords,stcmds,thchs}/train || exit 1;

  steps/align_fmllr.sh --cmd "$train_cmd" --nj 100 \
    data/train_all data/lang exp/tri3a exp/tri3a_ali || exit 1;
  steps/train_sat.sh --cmd "$train_cmd" 12000 190000 \
    data/train_all data/lang exp/tri3a_ali exp/tri4a || exit 1;
fi
