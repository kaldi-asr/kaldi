#!/bin/bash

# Copyright 2017  Vimal Manohar
# Apache 2.0

. cmd.sh
. path.sh 

stage=-1
train_stage=-10

. utils/parse_options.sh

set -o pipefail
exp=exp/semisup_50k

if [ ! -f data/train_sup/feats.scp ]; then
  echo "$0: Could not find data/train_sup/utt2spk"
  exit 1
fi

utils/subset_data_dir.sh --speakers data/train_sup 50000 data/train_sup50k || exit 1
utils/subset_data_dir.sh --shortest data/train_sup50k 25000 data/train_sup50k_short || exit 1
utils/subset_data_dir.sh --speakers data/train_sup50k 30000 data/train_sup50k_30k || exit 1;

steps/train_mono.sh --nj 10 --cmd "$train_cmd" \
  data/train_sup50k_short data/lang $exp/mono0a || exit 1

steps/align_si.sh --nj 30 --cmd "$train_cmd" \
  data/train_sup50k_30k data/lang $exp/mono0a $exp/mono0a_ali || exit 1

steps/train_deltas.sh --cmd "$train_cmd" \
  2500 20000 data/train_sup50k_30k data/lang $exp/mono0a_ali $exp/tri1 || exit 1

(utils/mkgraph.sh data/lang_test $exp/tri1 $exp/tri1/graph
 steps/decode.sh --nj 25 --cmd "$decode_cmd" --config conf/decode.config \
   $exp/tri1/graph data/dev $exp/tri1/decode_dev)&

steps/align_si.sh --nj 30 --cmd "$train_cmd" \
 data/train_sup50k_30k data/lang $exp/tri1 $exp/tri1_ali || exit 1;

steps/train_deltas.sh --cmd "$train_cmd" \
  2500 20000 data/train_sup50k_30k data/lang $exp/tri1_ali $exp/tri2 || exit 1

(utils/mkgraph.sh data/lang_test $exp/tri2 $exp/tri2/graph
 steps/decode.sh --nj 25 --cmd "$decode_cmd" --config conf/decode.config \
   $exp/tri2/graph data/dev $exp/tri2/decode_dev)&

steps/align_si.sh --nj 30 --cmd "$train_cmd" \
  data/train_sup50k data/lang $exp/tri2 $exp/tri2_ali || exit 1;

steps/train_lda_mllt.sh --cmd "$train_cmd" \
  4000 30000 data/train_sup50k data/lang $exp/tri2_ali $exp/tri3a || exit 1;

(utils/mkgraph.sh data/lang_test $exp/tri3a $exp/tri3a/graph
 steps/decode.sh --nj 25 --cmd "$decode_cmd" --config conf/decode.config \
   $exp/tri3a/graph data/dev $exp/tri3a/decode_dev)&

steps/align_fmllr.sh --nj 30 --cmd "$train_cmd" \
  data/train_sup50k data/lang $exp/tri3a $exp/tri3a_ali || exit 1;

steps/train_sat.sh --cmd "$train_cmd" \
  4000 50000 data/train_sup50k data/lang $exp/tri3a_ali $exp/tri4a || exit 1;

(
  utils/mkgraph.sh data/lang_test $exp/tri4a $exp/tri4a/graph
  steps/decode_fmllr.sh --nj 25 --cmd "$decode_cmd" --config conf/decode.config \
   $exp/tri4a/graph data/dev $exp/tri4a/decode_dev
)&

utils/combine_data.sh data/semisup50k_250k data/train_sup50k data/train_unsup250k || exit 1

local/semisup/chain/tuning/run_tdnn_50k.sh \
  --train-set train_sup50k \
  --nnet3-affix _semi50k_250k \
  --chain-affix _semi50k_250k \
  --gmm tri4a \
  --stage $stage --train-stage $train_stage \
  --exp $exp \
  --ivector-train-set semisup50k_250k || exit 1

local/semisup/chain/tuning/run_tdnn_oracle.sh \
  --train-set semisup50k_250k \
  --nnet3-affix _semi50k_250k \
  --chain-affix _semi50k_250k_oracle \
  --gmm tri4a \
  --stage 9 --train-stage $train_stage \
  --exp $exp \
  --ivector-train-set semisup50k_250k || exit 1
