#!/usr/bin/env bash

# Copyright 2017  Vimal Manohar
# Apache 2.0

# This script demonstrates semi-supervised training using 100 hours of 
# supervised data and 250 hours of unsupervised data.
# We assume the supervised data is in data/train_sup and unsupervised data
# is in data/train_unsup100k_250k. 
# For LM training, we only use the supervised set corresponding to 100 hours as 
# opposed to the case in run_50k.sh, where we included part of the 
# transcripts in data/train/text.
# This uses only 100 hours supervised set for i-vector extractor training, 
# which is different from run_50k.sh, which uses combined supervised + 
# unsupervised set.

. ./cmd.sh
. ./path.sh 

set -o pipefail
exp_root=exp/semisup_100k

stage=0

. utils/parse_options.sh

for f in data/train_sup/utt2spk data/train_unsup100k_250k/utt2spk \
  data/train_sup/text; do
  if [ ! -f $f ]; then
    echo "$0: Could not find $f"
    exit 1
  fi
done

###############################################################################
# Prepare the 100 hours supervised set and subsets for initial GMM training
###############################################################################

if [ $stage -le 0 ]; then
  utils/subset_data_dir.sh --shortest data/train_sup 100000 data/train_sup_100kshort
  utils/subset_data_dir.sh  data/train_sup_100kshort 10000 data/train_sup_10k
  utils/data/remove_dup_utts.sh 100 data/train_sup_10k data/train_sup_10k_nodup
  utils/subset_data_dir.sh --speakers data/train_sup 30000 data/train_sup_30k
fi

###############################################################################
# GMM system training using 100 hours supervised data
###############################################################################

if [ $stage -le 1 ]; then
  steps/train_mono.sh --nj 10 --cmd "$train_cmd" \
    data/train_sup_10k_nodup data/lang $exp_root/mono0a || exit 1
fi

if [ $stage -le 2 ]; then
  steps/align_si.sh --nj 30 --cmd "$train_cmd" \
    data/train_sup_30k data/lang $exp_root/mono0a $exp_root/mono0a_ali || exit 1

  steps/train_deltas.sh --cmd "$train_cmd" \
    2500 20000 data/train_sup_30k data/lang $exp_root/mono0a_ali $exp_root/tri1 || exit 1

  (utils/mkgraph.sh data/lang_test $exp_root/tri1 $exp_root/tri1/graph
   steps/decode.sh --nj 25 --cmd "$decode_cmd" --config conf/decode.config \
     $exp_root/tri1/graph data/dev $exp_root/tri1/decode_dev)&
fi

if [ $stage -le 3 ]; then
  steps/align_si.sh --nj 30 --cmd "$train_cmd" \
   data/train_sup_30k data/lang $exp_root/tri1 $exp_root/tri1_ali || exit 1;

  steps/train_lda_mllt.sh --cmd "$train_cmd" \
    2500 20000 data/train_sup_30k data/lang $exp_root/tri1_ali $exp_root/tri2 || exit 1;

  (utils/mkgraph.sh data/lang_test $exp_root/tri2 $exp_root/tri2/graph
   steps/decode.sh --nj 25 --cmd "$decode_cmd" --config conf/decode.config \
     $exp_root/tri2/graph data/dev $exp_root/tri2/decode_dev)&
fi

if [ $stage -le 4 ]; then
  steps/align_si.sh --nj 30 --cmd "$train_cmd" \
    data/train_sup data/lang $exp_root/tri2 $exp_root/tri2_ali || exit 1;

  steps/train_lda_mllt.sh --cmd "$train_cmd" \
     --splice-opts "--left-context=3 --right-context=3" \
     5000 40000 data/train_sup data/lang $exp_root/tri2_ali $exp_root/tri3a || exit 1;

  (
    utils/mkgraph.sh data/lang_test $exp_root/tri3a $exp_root/tri3a/graph || exit 1;
    steps/decode.sh --nj 25 --cmd "$decode_cmd" --config conf/decode.config \
     $exp_root/tri3a/graph data/dev $exp_root/tri3a/decode_dev || exit 1;
  )&
fi

if [ $stage -le 5 ]; then
  steps/align_fmllr.sh --nj 30 --cmd "$train_cmd" \
    data/train_sup data/lang $exp_root/tri3a $exp_root/tri3a_ali || exit 1;

  steps/train_sat.sh --cmd "$train_cmd" \
    5000 100000 data/train_sup data/lang $exp_root/tri3a_ali $exp_root/tri4a || exit 1;

  (
    utils/mkgraph.sh data/lang_test $exp_root/tri4a $exp_root/tri4a/graph
    steps/decode_fmllr.sh --nj 25 --cmd "$decode_cmd" --config conf/decode.config \
     $exp_root/tri4a/graph data/dev $exp_root/tri4a/decode_dev
  )&
fi

###############################################################################
# Prepare semi-supervised train set 
###############################################################################

if [ $stage -le 6 ]; then
  utils/combine_data.sh data/semisup100k_250k \
    data/train_sup data/train_unsup100k_250k || exit 1
fi

###############################################################################
# Train LM on the supervised set
###############################################################################

if [ $stage -le 7 ]; then
  if [ ! -f data/lang_test_poco_sup100k/G.fst ]; then
    local/fisher_train_lms_pocolm.sh \
      --text data/train_sup/text \
      --dir data/local/lm_sup100k

    local/fisher_create_test_lang.sh \
      --arpa-lm data/local/pocolm_sup100k/data/arpa/4gram_small.arpa.gz \
      --dir data/lang_test_poco_sup100k
  fi
fi

###############################################################################
# Prepare lang directories with UNK modeled using phone LM
###############################################################################

if [ $stage -le 8 ]; then
  local/run_unk_model.sh || exit 1

  for lang_dir in data/lang_test_poco_sup100k; do
    rm -r ${lang_dir}_unk 2>/dev/null || true
    cp -rT data/lang_unk ${lang_dir}_unk
    cp ${lang_dir}/G.fst ${lang_dir}_unk/G.fst
  done
fi

###############################################################################
# Train seed chain system using 100 hours supervised data.
# Here we train i-vector extractor on only the supervised set.
###############################################################################

if [ $stage -le 9 ]; then
  local/semisup/chain/run_tdnn.sh \
    --train-set train_sup \
    --ivector-train-set "" \
    --nnet3-affix "" --chain-affix "" \
    --tdnn-affix _1a --tree-affix bi_a \
    --gmm tri4a --exp-root $exp_root || exit 1

  # WER on dev                19.23
  # WER on test               19.01
  # Final train prob          -0.1224
  # Final valid prob          -0.1503
  # Final train prob (xent)   -1.6454
  # Final valid prob (xent)   -1.7107
fi

###############################################################################
# Semi-supervised training using 100 hours supervised data and 
# 250 hours unsupervised data. We use i-vector extractor, tree, lattices 
# and seed chain system from the previous stage.
###############################################################################

if [ $stage -le 10 ]; then
  local/semisup/chain/run_tdnn_100k_semisupervised.sh \
    --supervised-set train_sup \
    --unsupervised-set train_unsup100k_250k \
    --sup-chain-dir $exp_root/chain/tdnn_1a_sp \
    --sup-lat-dir $exp_root/chain/tri4a_train_sup_unk_lats \
    --sup-tree-dir $exp_root/chain/tree_bi_a \
    --ivector-root-dir $exp_root/nnet3 \
    --chain-affix "" \
    --tdnn-affix _semisup_1a \
    --exp-root $exp_root || exit 1

  # WER on dev                          18.70
  # WER on test                         18.18
  # Final output-0 train prob           -0.1345
  # Final output-0 valid prob           -0.1547
  # Final output-0 train prob (xent)    -1.3683
  # Final output-0 valid prob (xent)    -1.4077
  # Final output-1 train prob           -0.6856
  # Final output-1 valid prob           -0.6815
  # Final output-1 train prob (xent)    -1.1224
  # Final output-1 valid prob (xent)    -1.2218
fi

###############################################################################
# Oracle system trained on combined 350 hours including both supervised and 
# unsupervised sets. We use i-vector extractor, tree, and GMM trained
# on only the supervised for fair comparison to semi-supervised experiments.
###############################################################################

if [ $stage -le 11 ]; then
  local/semisup/chain/run_tdnn.sh \
    --train-set semisup100k_250k \
    --nnet3-affix "" --chain-affix "" \
    --common-treedir $exp_root/chain/tree_bi_a \
    --tdnn-affix 1a_oracle --nj 100 \
    --gmm tri4a --exp $exp_root \
    --stage 9 || exit 1

  # WER on dev                          16.97
  # WER on test                         17.03
  # Final output train prob             -0.1196
  # Final output valid prob             -0.1469
  # Final output train prob (xent)      -1.5487
  # Final output valid prob (xent)      -1.6360
fi
