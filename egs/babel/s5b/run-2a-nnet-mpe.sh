#!/bin/bash

. conf/common_vars.sh
. ./lang.conf

set -e
set -o pipefail
set -u

# Wait for cross-entropy training.
echo "Waiting till exp/tri6_nnet/.done exists...."
while [ ! -f exp/tri6_nnet/.done ]; do sleep 30; done
echo "...done waiting for exp/tri6_nnet/.done"

# Generate denominator lattices.
if [ ! -f exp/tri6_nnet_denlats/.done ]; then
  steps/nnet2/make_denlats.sh "${dnn_denlats_extra_opts[@]}" \
    --nj $train_nj --sub-split $train_nj \
    --transform-dir exp/tri5_ali \
    data/train data/lang exp/tri6_nnet exp/tri6_nnet_denlats || exit 1
 
  touch exp/tri6_nnet_denlats/.done
fi

# Generate alignment.
if [ ! -f exp/tri6_nnet_ali/.done ]; then
  steps/nnet2/align.sh --use-gpu yes \
    --cmd "$decode_cmd $dnn_parallel_opts" \
    --transform-dir exp/tri5_ali --nj $train_nj \
    data/train data/lang exp/tri6_nnet exp/tri6_nnet_ali || exit 1

  touch exp/tri6_nnet_ali/.done
fi

train_stage=-100
if [ ! -f exp/tri6_nnet_mpe/.done ]; then
  steps/nnet2/train_discriminative.sh \
    --stage $train_stage --cmd "$decode_cmd" \
    --learning-rate $dnn_mpe_learning_rate \
    --modify-learning-rates true \
    --last-layer-factor $dnn_mpe_last_layer_factor \
    --num-epochs 4 --cleanup true \
    --retroactive $dnn_mpe_retroactive \
    --transform-dir exp/tri5_ali \
    "${dnn_gpu_mpe_parallel_opts[@]}" data/train data/lang \
    exp/tri6_nnet_ali exp/tri6_nnet_denlats exp/tri6_nnet/final.mdl exp/tri6_nnet_mpe || exit 1

  touch exp/tri6_nnet_mpe/.done
fi
