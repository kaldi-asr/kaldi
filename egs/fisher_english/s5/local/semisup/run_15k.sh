#!/bin/bash

# Copyright 2017  Vimal Manohar
# Apache 2.0

. cmd.sh
. path.sh 

stage=-1
train_stage=-10

. utils/parse_options.sh

set -o pipefail
exp=exp/semisup_15k

false && {
utils/subset_data_dir.sh --speakers data/train_sup 15000 data/train_sup15k || exit 1
utils/subset_data_dir.sh --shortest data/train_sup15k 5000 data/train_sup15k_short || exit 1
utils/subset_data_dir.sh data/train_sup15k 7500 data/train_sup15k_half || exit 1

steps/train_mono.sh --nj 10 --cmd "$train_cmd" \
  data/train_sup15k_short data/lang $exp/mono0a || exit 1

steps/align_si.sh --nj 30 --cmd "$train_cmd" \
  data/train_sup15k_half data/lang $exp/mono0a $exp/mono0a_ali || exit 1

steps/train_deltas.sh --cmd "$train_cmd" \
  2000 10000 data/train_sup15k_half data/lang $exp/mono0a_ali $exp/tri1 || exit 1

(utils/mkgraph.sh data/lang_test $exp/tri1 $exp/tri1/graph
 steps/decode.sh --nj 25 --cmd "$decode_cmd" --config conf/decode.config \
   $exp/tri1/graph data/dev $exp/tri1/decode_dev)&

steps/align_si.sh --nj 30 --cmd "$train_cmd" \
 data/train_sup15k data/lang $exp/tri1 $exp/tri1_ali || exit 1;

steps/train_lda_mllt.sh --cmd "$train_cmd" \
  2500 15000 data/train_sup15k data/lang $exp/tri1_ali $exp/tri2 || exit 1;

(utils/mkgraph.sh data/lang_test $exp/tri2 $exp/tri2/graph
 steps/decode.sh --nj 25 --cmd "$decode_cmd" --config conf/decode.config \
   $exp/tri2/graph data/dev $exp/tri2/decode_dev)&

steps/align_si.sh --nj 30 --cmd "$train_cmd" \
  data/train_sup15k data/lang $exp/tri2 $exp/tri2_ali || exit 1;

steps/train_sat.sh --cmd "$train_cmd" \
  2500 15000 data/train_sup15k data/lang $exp/tri2_ali $exp/tri3 || exit 1;

(
  utils/mkgraph.sh data/lang_test $exp/tri3 $exp/tri3/graph
  steps/decode_fmllr.sh --nj 25 --cmd "$decode_cmd" --config conf/decode.config \
   $exp/tri3/graph data/dev $exp/tri3/decode_dev
)&

utils/combine_data.sh data/semisup15k_250k data/train_sup15k data/train_unsup250k || exit 1
}

local/semisup/chain/tuning/run_tdnn_11k.sh \
  --train-set train_sup15k \
  --nnet3-affix _semi15k_250k \
  --chain-affix _semi15k_250k \
  --stage $stage --train-stage $train_stage \
  --exp $exp \
  --ivector-train-set semisup15k_250k || exit 1
