#!/bin/bash

# Daniel Povey (Johns Hopkins University) 's style DNN training
#
# Prepared by Ricky Chan Ho Yin (Hong Kong University of Science and Technology)
#
# Apache License, 2.0

. cmd.sh

. path.sh

ulimit -u 10000

(
 steps/nnet2/train_tanh.sh \
   --mix-up 8000 \
   --initial-learning-rate 0.01 --final-learning-rate 0.001 \
   --num-hidden-layers 6 --hidden-layer-dim 1024 \
   --cmd "$decode_cmd" \
   data/train data/lang exp/tri5a_ali_dt100k exp/nnet_tanh_6l || exit 1

steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 2 --transform-dir exp/tri5a/decode_eval exp/tri5a/graph data/eval exp/nnet_tanh_6l/decode_eval &
steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 2 --transform-dir exp/tri5a/decode_eval_closelm exp/tri5a/graph_closelm data/eval exp/nnet_tanh_6l/decode_eval_closelm &

steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 2 --config conf/decode_wide.config --transform-dir exp/tri5a/decode_eval exp/tri5a/graph data/eval exp/nnet_tanh_6l/decode_wide_eval &
steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 2 --config conf/decode_wide.config --transform-dir exp/tri5a/decode_eval_closelm exp/tri5a/graph_closelm data/eval exp/nnet_tanh_6l/decode_wide_eval_closelm &
wait


local/ext/score.sh data/eval exp/tri5a/graph exp/nnet_tanh_6l/decode_eval
local/ext/score.sh data/eval exp/tri5a/graph_closelm exp/nnet_tanh_6l/decode_eval_closelm

local/ext/score.sh data/eval exp/tri5a/graph exp/nnet_tanh_6l/decode_wide_eval
local/ext/score.sh data/eval exp/tri5a/graph_closelm exp/nnet_tanh_6l/decode_wide_eval_closelm

)

