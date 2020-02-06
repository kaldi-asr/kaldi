#!/usr/bin/env bash

# Copyright 2019 Microsoft Corporation (authors: Xingyu Na)
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

test_sets="aishell aidatatang magicdata thchs"
corpus_lm=false   # interpolate with corpus lm

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
  local/prepare_dict.sh || exit 1;
fi

if [ $stage -le 3 ]; then
  # train LM using transcription
  local/train_lms.sh || exit 1;
  if $corpus_lm; then
    for c in $test_sets; do
      local/train_corpus_lm.sh $c data/local/lm/3gram-mincount/lm_unpruned.gz
    done
  fi
fi

if [ $stage -le 4 ]; then
  # prepare LM
  utils/prepare_lang.sh data/local/dict "<UNK>" data/local/lang data/lang || exit 1;
  utils/format_lm.sh data/lang data/local/lm/3gram-mincount/lm_unpruned.gz \
    data/local/dict/lexicon.txt data/lang_combined_tg || exit 1;
  if $corpus_lm; then
    for c in $test_sets; do
      utils/format_lm.sh data/lang data/local/lm/$c/lm_interp.gz \
        data/local/dict/lexicon.txt data/lang_${c}_tg || exit 1;
    done
  fi
fi

if [ $stage -le 5 ]; then
  # make features
  mfccdir=mfcc
  corpora="aidatatang aishell magicdata primewords stcmds thchs"
  for c in $corpora; do
    (
      steps/make_mfcc_pitch_online.sh --cmd "$train_cmd" --nj 20 \
        data/$c/train exp/make_mfcc/$c/train $mfccdir/$c || exit 1;
      steps/compute_cmvn_stats.sh data/$c/train \
        exp/make_mfcc/$c/train $mfccdir/$c || exit 1;
    ) &
  done
  wait
fi

if [ $stage -le 6 ]; then
  # make test features
  mfccdir=mfcc
  for c in $test_sets; do
    (
      steps/make_mfcc_pitch_online.sh --cmd "$train_cmd" --nj 10 \
        data/$c/test exp/make_mfcc/$c/test $mfccdir/$c || exit 1;
      steps/compute_cmvn_stats.sh data/$c/test \
        exp/make_mfcc/$c/test $mfccdir/$c || exit 1;
    ) &
  done
  wait
fi

if [ $stage -le 7 ]; then
  # train mono and tri1a using aishell(~120k)
  # mono has been used in aishell recipe, so no test
  steps/train_mono.sh --boost-silence 1.25 --nj 20 --cmd "$train_cmd" \
    data/aishell/train data/lang exp/mono || exit 1;

  steps/align_si.sh --boost-silence 1.25 --nj 20 --cmd "$train_cmd" \
    data/aishell/train data/lang exp/mono exp/mono_ali || exit 1;
  steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" 2500 20000 \
    data/aishell/train data/lang exp/mono_ali exp/tri1a || exit 1;
fi

if [ $stage -le 8 ]; then
  # train tri1b using aishell + primewords + stcmds + thchs (~280k)
  utils/combine_data.sh data/train_280k \
    data/{aishell,primewords,stcmds,thchs}/train || exit 1;

  steps/align_si.sh --boost-silence 1.25 --nj 40 --cmd "$train_cmd" \
    data/train_280k data/lang exp/tri1a exp/tri1a_280k_ali || exit 1;
  steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" 4500 36000 \
    data/train_280k data/lang exp/tri1a_280k_ali exp/tri1b || exit 1;
fi

if [ $stage -le 9 ]; then
  # test tri1b
  utils/mkgraph.sh data/lang_combined_tg exp/tri1b exp/tri1b/graph_tg || exit 1;
  for c in $test_sets; do
    (
      steps/decode.sh --cmd "$decode_cmd" --config conf/decode.config --nj 10 \
        exp/tri1b/graph_tg data/$c/test exp/tri1b/decode_${c}_test_tg || exit 1;
      if $corpus_lm; then
        $mkgraph_cmd exp/tri1b/log/mkgraph.$c.log \
          utils/mkgraph.sh data/lang_${c}_tg exp/tri1b exp/tri1b/graph_$c || exit 1;
        steps/decode.sh --cmd "$decode_cmd" --config conf/decode.config --nj 10 \
          exp/tri1b/graph_$c data/$c/test exp/tri1b/decode_${c}_test_clm || exit 1;
      fi
    ) &
  done
  wait
fi

if [ $stage -le 10 ]; then
  # train tri2a using train_280k
  steps/align_si.sh --boost-silence 1.25 --nj 40 --cmd "$train_cmd" \
    data/train_280k data/lang exp/tri1b exp/tri1b_280k_ali || exit 1;
  steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" 5500 90000 \
    data/train_280k data/lang exp/tri1b_280k_ali exp/tri2a || exit 1;
fi

if [ $stage -le 11 ]; then
  # test tri2a
  utils/mkgraph.sh data/lang_combined_tg exp/tri2a exp/tri2a/graph_tg || exit 1;
  for c in $test_sets; do
    (
      steps/decode.sh --cmd "$decode_cmd" --config conf/decode.config --nj 10 \
        exp/tri2a/graph_tg data/$c/test exp/tri2a/decode_${c}_test_tg || exit 1;
      if $corpus_lm; then
        $mkgraph_cmd exp/tri2a/log/mkgraph.$c.log \
          utils/mkgraph.sh data/lang_${c}_tg exp/tri2a exp/tri2a/graph_$c || exit 1;
        steps/decode.sh --cmd "$decode_cmd" --config conf/decode.config --nj 10 \
          exp/tri2a/graph_$c data/$c/test exp/tri2a/decode_${c}_test_clm || exit 1;
      fi
    ) &
  done
  wait
fi

if [ $stage -le 12 ]; then
  # train tri3a using aidatatang + aishell + primewords + stcmds + thchs (~440k)
  utils/combine_data.sh data/train_440k \
    data/{aidatatang,aishell,primewords,stcmds,thchs}/train || exit 1;

  steps/align_si.sh --boost-silence 1.25 --nj 60 --cmd "$train_cmd" \
    data/train_440k data/lang exp/tri2a exp/tri2a_440k_ali || exit 1;
  steps/train_lda_mllt.sh --cmd "$train_cmd" 7000 110000 \
    data/train_440k data/lang exp/tri2a_440k_ali exp/tri3a || exit 1;
fi

if [ $stage -le 13 ]; then
  # test tri3a
  utils/mkgraph.sh data/lang_combined_tg exp/tri3a exp/tri3a/graph_tg || exit 1;
  for c in $test_sets; do
    (
      steps/decode.sh --cmd "$decode_cmd" --config conf/decode.config --nj 10 \
        exp/tri3a/graph_tg data/$c/test exp/tri3a/decode_${c}_test_tg || exit 1;
      if $corpus_lm; then
        $mkgraph_cmd exp/tri3a/log/mkgraph.$c.log \
          utils/mkgraph.sh data/lang_${c}_tg exp/tri3a exp/tri3a/graph_$c || exit 1;
        steps/decode.sh --cmd "$decode_cmd" --config conf/decode.config --nj 10 \
          exp/tri3a/graph_$c data/$c/test exp/tri3a/decode_${c}_test_clm || exit 1;
      fi
    ) &
  done
  wait
fi

if [ $stage -le 14 ]; then
  # train tri4a using all
  utils/combine_data.sh data/train_all \
    data/{aidatatang,aishell,magicdata,primewords,stcmds,thchs}/train || exit 1;

  steps/align_fmllr.sh --cmd "$train_cmd" --nj 100 \
    data/train_all data/lang exp/tri3a exp/tri3a_ali || exit 1;
  steps/train_sat.sh --cmd "$train_cmd" 12000 190000 \
    data/train_all data/lang exp/tri3a_ali exp/tri4a || exit 1;
fi

if [ $stage -le 15 ]; then
  # test tri4a
  utils/mkgraph.sh data/lang_combined_tg exp/tri4a exp/tri4a/graph_tg || exit 1;
  for c in $test_sets; do
    (
      steps/decode_fmllr.sh --cmd "$decode_cmd" --config conf/decode.config --nj 10 \
        exp/tri4a/graph_tg data/$c/test exp/tri4a/decode_${c}_test_tg || exit 1;
      if $corpus_lm; then
        $mkgraph_cmd exp/tri4a/log/mkgraph.$c.log \
          utils/mkgraph.sh data/lang_${c}_tg exp/tri4a exp/tri4a/graph_$c || exit 1;
        steps/decode_fmllr.sh --cmd "$decode_cmd" --config conf/decode.config --nj 10 \
          exp/tri4a/graph_$c data/$c/test exp/tri4a/decode_${c}_test_clm || exit 1;
      fi
    ) &
  done
  wait
fi

if [ $stage -le 16 ]; then
  # run clean and retrain
  local/run_cleanup_segmentation.sh --test-sets "$test_sets" --corpus-lm $corpus_lm
fi

if [ $stage -le 17 ]; then
  # collect GMM test results
  if ! $corpus_lm; then
    for c in $test_sets; do
      echo "$c test set results"
      for x in exp/*/decode_${c}*_tg; do
        grep WER $x/cer_* | utils/best_wer.sh
      done
      echo ""
    done
  else
    # collect corpus LM results
    for c in $test_sets; do
      echo "$c test set results"
      for x in exp/*/decode_${c}*_clm; do
        grep WER $x/cer_* | utils/best_wer.sh
      done
      echo ""
    done
  fi
fi

exit 0;

# chain modeling script
local/chain/run_cnn_tdnn.sh --test-sets "$test_sets"
for c in $test_sets; do
  for x in exp/chain_cleaned/*/decode_${c}*_tg; do
    grep WER $x/cer_* | utils/best_wer.sh
  done
done
