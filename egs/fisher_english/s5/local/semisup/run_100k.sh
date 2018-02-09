#!/bin/bash

# Copyright 2017  Vimal Manohar
# Apache 2.0

# This script demonstrates semi-supervised training using 100 hours of 
# supervised data and 250 hours of unsupervised data.
# We assume the supervised data is in data/train_sup and unsupervised data
# is in data/train_unsup100k_250k. 
# For LM training, we only use the supervised set corresponding to 100 hours as 
# opposed to the case in run_50k.sh, where we included part of the 
# transcripts in data/train/text.

. ./cmd.sh
. ./path.sh 

set -o pipefail
exp_root=exp/semisup_100k

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

utils/subset_data_dir.sh --shortest data/train_sup 100000 data/train_sup_100kshort
utils/subset_data_dir.sh  data/train_sup_100kshort 10000 data/train_sup_10k
utils/data/remove_dup_utts.sh 100 data/train_sup_10k data/train_sup_10k_nodup
utils/subset_data_dir.sh --speakers data/train_sup 30000 data/train_sup_30k

###############################################################################
# GMM system training using 100 hours supervised data
###############################################################################

steps/train_mono.sh --nj 10 --cmd "$train_cmd" \
  data/train_sup_10k_nodup data/lang $exp_root/mono0a || exit 1

steps/align_si.sh --nj 30 --cmd "$train_cmd" \
  data/train_sup_30k data/lang $exp_root/mono0a $exp_root/mono0a_ali || exit 1

steps/train_deltas.sh --cmd "$train_cmd" \
  2500 20000 data/train_sup_30k data/lang $exp_root/mono0a_ali $exp_root/tri1 || exit 1

(utils/mkgraph.sh data/lang_test $exp_root/tri1 $exp_root/tri1/graph
 steps/decode.sh --nj 25 --cmd "$decode_cmd" --config conf/decode.config \
   $exp_root/tri1/graph data/dev $exp_root/tri1/decode_dev)&

steps/align_si.sh --nj 30 --cmd "$train_cmd" \
 data/train_sup_30k data/lang $exp_root/tri1 $exp_root/tri1_ali || exit 1;

steps/train_lda_mllt.sh --cmd "$train_cmd" \
  2500 20000 data/train_sup_30k data/lang $exp_root/tri1_ali $exp_root/tri2 || exit 1;

(utils/mkgraph.sh data/lang_test $exp_root/tri2 $exp_root/tri2/graph
 steps/decode.sh --nj 25 --cmd "$decode_cmd" --config conf/decode.config \
   $exp_root/tri2/graph data/dev $exp_root/tri2/decode_dev)&

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

steps/align_fmllr.sh --nj 30 --cmd "$train_cmd" \
  data/train_sup data/lang $exp_root/tri3a $exp_root/tri3a_ali || exit 1;

steps/train_sat.sh --cmd "$train_cmd" \
  5000 100000 data/train_sup data/lang $exp_root/tri3a_ali $exp_root/tri4a || exit 1;

(
  utils/mkgraph.sh data/lang_test $exp_root/tri4a $exp_root/tri4a/graph
  steps/decode_fmllr.sh --nj 25 --cmd "$decode_cmd" --config conf/decode.config \
   $exp_root/tri4a/graph data/dev $exp_root/tri4a/decode_dev
)&

###############################################################################
# Prepare semi-supervised train set 
###############################################################################

utils/combine_data.sh data/semisup100k_250k \
  data/train_sup data/train_unsup100k_250k || exit 1

###############################################################################
# Train LM on the supervised set
###############################################################################

if [ ! -f data/lang_test_poco_sup100k/G.fst ]; then
  local/fisher_train_lms_pocolm.sh \
    --text data/train_sup/text \
    --dir data/local/lm_sup100k

  local/fisher_create_test_lang.sh \
    --arpa-lm data/local/pocolm_sup100k/data/arpa/4gram_small.arpa.gz \
    --dir data/lang_test_poco_sup100k
fi

###############################################################################
# Prepare lang directories with UNK modeled using phone LM
###############################################################################

local/run_unk_model.sh || exit 1

for lang_dir in data/lang_test_poco_sup100k; do
  rm -r ${lang_dir}_unk 2>/dev/null || true
  cp -rT data/lang_unk ${lang_dir}_unk
  cp ${lang_dir}/G.fst ${lang_dir}_unk/G.fst
done

###############################################################################
# Train seed chain system using 100 hours supervised data.
# Here we train i-vector extractor on only the supervised set.
###############################################################################

local/semisup/chain/run_tdnn.sh \
  --train-set train_sup \
  --ivector-train-set "" \
  --nnet3-affix "" --chain-affix "" \
  --tdnn-affix 1a --tree-affix bi_a \
  --gmm tri4a --exp-root $exp_root || exit 1

###############################################################################
# Semi-supervised training using 100 hours supervised data and 
# 250 hours unsupervised data. We use i-vector extractor, tree, lattices 
# and seed chain system from the previous stage.
###############################################################################

local/semisup/chain/run_tdnn_100k_semisupervised.sh \
  --supervised-set train_sup \
  --unsupervised-set train_unsup100k_250k \
  --sup-chain-dir $exp_root/chain/tdnn_1a_sp \
  --sup-lat-dir $exp_root/chain/tri4a_train_sup_unk_lats \
  --sup-tree-dir $exp_root/chain/tree_bi_a \
  --ivector-root-dir $exp_root/nnet3 \
  --chain-affix "" \
  --tdnn-affix _semisup_1a \
  --exp $exp_root --stage 0 || exit 1

###############################################################################
# Oracle system trained on combined 350 hours including both supervised and 
# unsupervised sets. We use i-vector extractor, tree, and GMM trained
# on only the supervised for fair comparison to semi-supervised experiments.
###############################################################################

local/semisup/chain/run_tdnn.sh \
  --train-set semisup100k_250k \
  --nnet3-affix "" --chain-affix "" \
  --common-treedir $exp_root/chain/tree_bi_a \
  --tdnn-affix 1a_oracle \
  --gmm tri4a --exp $exp_root \
  --stage 9 || exit 1
