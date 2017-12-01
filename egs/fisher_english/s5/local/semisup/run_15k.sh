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

for f in data/train_sup/utt2spk data/train_unsup250k/utt2spk ]; do
  if [ ! -f $f ]; then
    echo "$0: Could not find $f"
    exit 1
  fi
done

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

mkdir -p data/local/pocolm_ex250k

utils/filter_scp.pl --exclude data/train_unsup250k/utt2spk \
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

local/run_unk_model.sh --lang-dirs "data/lang_test_poco_ex250k_big data/lang_test_poco_ex250k" || exit 1

local/semisup/chain/tuning/run_tdnn_11k.sh \
  --train-set train_sup15k \
  --nnet3-affix _semi15k_250k \
  --chain-affix _semi15k_250k \
  --stage $stage --train-stage $train_stage \
  --exp $exp \
  --ivector-train-set semisup15k_250k || exit 1

local/semisup/chain/tuning/run_tdnn_oracle.sh \
  --train-set semisup15k_250k \
  --nnet3-affix _semi15k_250k \
  --chain-affix _semi15k_250k_oracle \
  --gmm tri3 \
  --stage 9 --train-stage $train_stage \
  --exp $exp \
  --ivector-train-set semisup15k_250k || exit 1
