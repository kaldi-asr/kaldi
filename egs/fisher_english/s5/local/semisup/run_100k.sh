#!/bin/bash

# Copyright 2017  Vimal Manohar
# Apache 2.0

. cmd.sh
. path.sh 

stage=-1
train_stage=-10

. utils/parse_options.sh

set -o pipefail
exp=exp/semisup_100k

true && {
utils/subset_data_dir.sh --shortest data/train_sup 100000 data/train_sup_100kshort
utils/subset_data_dir.sh  data/train_sup_100kshort 10000 data/train_sup_10k
utils/data/remove_dup_utts.sh 100 data/train_sup_10k data/train_sup_10k_nodup
utils/subset_data_dir.sh --speakers data/train_sup 30000 data/train_sup_30k

steps/train_mono.sh --nj 10 --cmd "$train_cmd" \
  data/train_sup_10k_nodup data/lang $exp/mono0a || exit 1

steps/align_si.sh --nj 30 --cmd "$train_cmd" \
  data/train_sup_30k data/lang $exp/mono0a $exp/mono0a_ali || exit 1

steps/train_deltas.sh --cmd "$train_cmd" \
  2500 20000 data/train_sup_30k data/lang $exp/mono0a_ali $exp/tri1 || exit 1

(utils/mkgraph.sh data/lang_test $exp/tri1 $exp/tri1/graph
 steps/decode.sh --nj 25 --cmd "$decode_cmd" --config conf/decode.config \
   $exp/tri1/graph data/dev $exp/tri1/decode_dev)&

steps/align_si.sh --nj 30 --cmd "$train_cmd" \
 data/train_sup_30k data/lang $exp/tri1 $exp/tri1_ali || exit 1;

steps/train_lda_mllt.sh --cmd "$train_cmd" \
  2500 20000 data/train_sup_30k data/lang $exp/tri1_ali $exp/tri2 || exit 1;

(utils/mkgraph.sh data/lang_test $exp/tri2 $exp/tri2/graph
 steps/decode.sh --nj 25 --cmd "$decode_cmd" --config conf/decode.config \
   $exp/tri2/graph data/dev $exp/tri2/decode_dev)&

steps/align_si.sh --nj 30 --cmd "$train_cmd" \
  data/train_sup data/lang $exp/tri2 $exp/tri2_ali || exit 1;

steps/train_lda_mllt.sh --cmd "$train_cmd" \
   --splice-opts "--left-context=3 --right-context=3" \
   5000 40000 data/train_sup data/lang $exp/tri2_ali $exp/tri3a || exit 1;

(
  utils/mkgraph.sh data/lang_test $exp/tri3a $exp/tri3a/graph || exit 1;
  steps/decode.sh --nj 25 --cmd "$decode_cmd" --config conf/decode.config \
   $exp/tri3a/graph data/dev $exp/tri3a/decode_dev || exit 1;
)&

steps/align_fmllr.sh --nj 30 --cmd "$train_cmd" \
  data/train_sup data/lang $exp/tri3a $exp/tri3a_ali || exit 1;

steps/train_sat.sh --cmd "$train_cmd" \
  5000 100000 data/train_sup data/lang $exp/tri3a_ali $exp/tri4a || exit 1;

(
  utils/mkgraph.sh data/lang_test $exp/tri4a $exp/tri4a/graph
  steps/decode_fmllr.sh --nj 25 --cmd "$decode_cmd" --config conf/decode.config \
   $exp/tri4a/graph data/dev $exp/tri4a/decode_dev
)&

utils/copy_data_dir.sh data/train_unsup250k data/train_unsup100k_250k
utils/combine_data.sh data/semisup100k_250k data/train_sup \
  data/train_unsup250k || exit 1

local/semisup/chain/tuning/run_tdnn_100k.sh \
  --train-set train_sup \
  --stage $stage --train-stage $train_stage \
  --exp $exp \
  --ivector-train-set train_sup || exit 1
}

false && local/semisup/chain/tuning/run_tdnn_oracle.sh \
  --train-set semisup15k_250k \
  --nnet3-affix _semi15k_250k \
  --chain-affix _semi15k_250k_oracle \
  --stage 9 --train-stage $train_stage \
  --exp $exp \
  --ivector-train-set semisup15k_250k || exit 1

