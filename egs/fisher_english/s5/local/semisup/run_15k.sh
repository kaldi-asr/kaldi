#!/bin/bash

# Copyright 2017  Vimal Manohar
# Apache 2.0

# This script demonstrates semi-supervised training using 15 hours of 
# supervised data and 250 hours of unsupervised data.
# We assume the supervised data is in data/train_sup and unsupervised data
# is in data/train_unsup100k_250k. 
# Further, for LM training we assume there is data/train/text, from which
# we will exclude the utterances contained in the unsupervised set.

. ./cmd.sh
. ./path.sh 

set -o pipefail
exp_root=exp/semisup_15k

for f in data/train_sup/utt2spk data/train_unsup100k_250k/utt2spk \
  data/train/text; do
  if [ ! -f $f ]; then
    echo "$0: Could not find $f"
    exit 1
  fi
done

###############################################################################
# Prepare the 15 hours supervised set and subsets for initial GMM training
###############################################################################

utils/subset_data_dir.sh --speakers data/train_sup 15000 data/train_sup15k || exit 1
utils/subset_data_dir.sh --shortest data/train_sup15k 5000 data/train_sup15k_short || exit 1
utils/subset_data_dir.sh data/train_sup15k 7500 data/train_sup15k_half || exit 1

###############################################################################
# GMM system training using 15 hours supervised data
###############################################################################

steps/train_mono.sh --nj 10 --cmd "$train_cmd" \
  data/train_sup15k_short data/lang $exp_root/mono0a || exit 1

steps/align_si.sh --nj 30 --cmd "$train_cmd" \
  data/train_sup15k_half data/lang $exp_root/mono0a $exp_root/mono0a_ali || exit 1

steps/train_deltas.sh --cmd "$train_cmd" \
  2000 10000 data/train_sup15k_half data/lang $exp_root/mono0a_ali $exp_root/tri1 || exit 1

(utils/mkgraph.sh data/lang_test $exp_root/tri1 $exp_root/tri1/graph
 steps/decode.sh --nj 25 --cmd "$decode_cmd" --config conf/decode.config \
   $exp_root/tri1/graph data/dev $exp_root/tri1/decode_dev)&

steps/align_si.sh --nj 30 --cmd "$train_cmd" \
 data/train_sup15k data/lang $exp_root/tri1 $exp_root/tri1_ali || exit 1;

steps/train_lda_mllt.sh --cmd "$train_cmd" \
  2500 15000 data/train_sup15k data/lang $exp_root/tri1_ali $exp_root/tri2 || exit 1;

(utils/mkgraph.sh data/lang_test $exp_root/tri2 $exp_root/tri2/graph
 steps/decode.sh --nj 25 --cmd "$decode_cmd" --config conf/decode.config \
   $exp_root/tri2/graph data/dev $exp_root/tri2/decode_dev)&

steps/align_si.sh --nj 30 --cmd "$train_cmd" \
  data/train_sup15k data/lang $exp_root/tri2 $exp_root/tri2_ali || exit 1;

steps/train_sat.sh --cmd "$train_cmd" \
  2500 15000 data/train_sup15k data/lang $exp_root/tri2_ali $exp_root/tri3 || exit 1;

(
  utils/mkgraph.sh data/lang_test $exp_root/tri3 $exp_root/tri3/graph
  steps/decode_fmllr.sh --nj 25 --cmd "$decode_cmd" --config conf/decode.config \
   $exp_root/tri3/graph data/dev $exp_root/tri3/decode_dev
)&

###############################################################################
# Prepare semi-supervised train set 
###############################################################################

utils/combine_data.sh data/semisup15k_100k_250k \
  data/train_sup15k data/train_unsup100k_250k || exit 1

###############################################################################
# Train LM on all the text in data/train/text, but excluding the 
# utterances in the unsupervised set
###############################################################################

mkdir -p data/local/pocolm_ex250k

utils/filter_scp.pl --exclude data/train_unsup100k_250k/utt2spk \
  data/train/text > data/local/pocolm_ex250k/text.tmp

if [ ! -f data/lang_test_poco_ex250k_big/G.carpa ]; then
  local/fisher_train_lms_pocolm.sh \
    --text data/local/pocolm_ex250k/text.tmp \
    --dir data/local/pocolm_ex250k

  local/fisher_create_test_lang.sh \
    --arpa-lm data/local/pocolm_ex250k/data/arpa/4gram_small.arpa.gz \
    --dir data/lang_test_poco_ex250k

  utils/build_const_arpa_lm.sh \
    data/local/pocolm_ex250k/data/arpa/4gram_big.arpa.gz \
    data/lang_test_poco_ex250k data/lang_test_poco_ex250k_big
fi

###############################################################################
# Prepare lang directories with UNK modeled using phone LM
###############################################################################

local/run_unk_model.sh || exit 1

for lang_dir in data/lang_test_poco_ex250k; do
  rm -r ${lang_dir}_unk ${lang_dir}_unk_big 2>/dev/null || true
  cp -rT data/lang_unk ${lang_dir}_unk
  cp ${lang_dir}/G.fst ${lang_dir}_unk/G.fst
  cp -rT data/lang_unk ${lang_dir}_unk_big
  cp ${lang_dir}_big/G.carpa ${lang_dir}_unk_big/G.carpa; 
done

###############################################################################
# Train seed chain system using 50 hours supervised data.
# Here we train i-vector extractor on combined supervised and unsupervised data
###############################################################################

local/semisup/chain/run_tdnn.sh \
  --train-set train_sup15k \
  --ivector-train-set semisup15k_100k_250k \
  --nnet3-affix _semi15k_100k_250k \
  --chain-affix _semi15k_100k_250k \
  --tdnn-affix 1a --tree-affix bi_a \
  --hidden-dim 500 \
  --gmm tri3 --exp-root $exp_root || exit 1

###############################################################################
# Semi-supervised training using 15 hours supervised data and 
# 250 hours unsupervised data. We use i-vector extractor, tree, lattices 
# and seed chain system from the previous stage.
###############################################################################

local/semisup/chain/run_tdnn_50k_semisupervised.sh \
  --supervised-set train_sup15k \
  --unsupervised-set train_unsup100k_250k \
  --sup-chain-dir $exp_root/chain_semi15k_100k_250k/tdnn_1a_sp \
  --sup-lat-dir $exp_root/chain_semi15k_100k_250k/tri3_train_sup15k_unk_lats \
  --sup-tree-dir $exp_root/chain_semi15k_100k_250k/tree_bi_a \
  --ivector-root-dir $exp_root/nnet3_semi15k_100k_250k \
  --chain-affix _semi15k_100k_250k \
  --tdnn-affix _semisup_1a \
  --exp-root $exp_root --stage 0 || exit 1

###############################################################################
# Oracle system trained on combined 300 hours including both supervised and 
# unsupervised sets. We use i-vector extractor, tree, and GMM trained
# on only the supervised for fair comparison to semi-supervised experiments.
###############################################################################

local/semisup/chain/run_tdnn.sh \
  --train-set semisup15k_100k_250k \
  --nnet3-affix _semi15k_100k_250k \
  --chain-affix _semi15k_100k_250k \
  --common-treedir $exp_root/chain_semi15k_100k_250k/tree_bi_a \
  --tdnn-affix 1a_oracle \
  --gmm tri3 --exp-root $exp_root \
  --stage 9 || exit 1
