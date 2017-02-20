#!/bin/bash


. conf/common_vars.sh
. ./lang.conf

modeldir=exp/tri6_nnet

. ./utils/parse_options.sh
set -e
set -o pipefail
set -u

# Wait for cross-entropy training.
echo "Waiting till ${modeldir}/.done exists...."
while [ ! -f $modeldir/.done ]; do sleep 30; done
echo "...done waiting for ${modeldir}/.done"

# Generate denominator lattices.
if [ ! -f exp/tri6_nnet_denlats/.done ]; then
  steps/nnet2/make_denlats.sh --cmd "$decode_cmd" \
    --nj $train_nj --sub-split $train_nj \
    "${dnn_denlats_extra_opts[@]}" \
    --transform-dir exp/tri5_ali \
    data/train data/langp/tri5_ali ${modeldir} exp/tri6_nnet_denlats || exit 1

  touch exp/tri6_nnet_denlats/.done
fi

# Generate alignment.
if [ ! -f exp/tri6_nnet_ali/.done ]; then
  steps/nnet2/align.sh --use-gpu yes \
    --cmd "$decode_cmd $dnn_parallel_opts" \
    --transform-dir exp/tri5_ali --nj $train_nj \
    data/train data/langp/tri5_ali ${modeldir} exp/tri6_nnet_ali || exit 1

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
    "${dnn_gpu_mpe_parallel_opts[@]}" data/train data/langp/tri5_ali/ \
    exp/tri6_nnet_ali exp/tri6_nnet_denlats ${modeldir}/final.mdl exp/tri6_nnet_mpe || exit 1

  touch exp/tri6_nnet_mpe/.done
fi
