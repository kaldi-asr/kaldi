#!/bin/bash

. cmd.sh

# Build a SAT system on the 30k data.
steps/align_fmllr.sh --nj 30 --cmd "$train_cmd" \
  data/train_30k_nodup data/lang exp/tri3a exp/tri3a_ali_30k_nodup 

steps/train_sat.sh  --cmd "$train_cmd" \
  2500 20000 data/train_30k_nodup data/lang exp/tri3a_ali_30k_nodup exp/tri4b

(
  utils/mkgraph.sh data/lang_test exp/tri4b exp/tri4b/graph
  steps/decode_fmllr.sh --nj 30 --cmd "$decode_cmd" --config conf/decode.config \
   exp/tri4b/graph data/train_dev exp/tri4b/decode_train_dev
)&

# Use the alignments already present in tri4b/.
# Note: this doesn't rebuild the tree, you can use the graph from tri4b/.
steps/train_nnet_cpu.sh --num-parameters 4000000 --samples_per_iteration 800000 \
   --cmd "$decode_cmd" --parallel-opts "-pe smp 16" \
  data/train_30k_nodup data/lang exp/tri4b exp/tri5b_nnet

steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 30 \
  --config conf/decode.config --transform-dir exp/tri4b/decode_train_dev \
  exp/tri4b/graph data/train_dev exp/tri5b_nnet/decode_train_dev

for iter in 3 4; do
 steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 30 \
  --iter $iter --config conf/decode.config --transform-dir exp/tri4b/decode_train_dev \
   exp/tri4b/graph data/train_dev exp/tri5b_nnet/decode_train_dev_it$iter
done

# as above but with 3, not 2, hidden layers.
steps/train_nnet_cpu.sh --num_hidden_layers 3 \
   --num-parameters 6000000 --samples_per_iteration 800000 \
   --cmd "$decode_cmd" --parallel-opts "-pe smp 16" \
  data/train_30k_nodup data/lang exp/tri4b exp/tri5b1_nnet

steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 30 \
  --config conf/decode.config --transform-dir exp/tri4b/decode_train_dev \
  exp/tri4b/graph data/train_dev exp/tri5b1_nnet/decode_train_dev

steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 30 \
  --beam 13.0 --transform-dir exp/tri4b/decode_train_dev \
  exp/tri4b/graph data/train_dev exp/tri5b1_nnet/decode_train_dev_wide


steps/train_nnet_cpu.sh --stage 5 --num-iters 15 --num_hidden_layers 3 \
   --num-parameters 6000000 --samples_per_iteration 800000 \
   --cmd "$decode_cmd" --parallel-opts "-pe smp 16" \
  data/train_30k_nodup data/lang exp/tri4b exp/tri5b1_nnet

# using more validation utts and more minibatches per phase.
steps/train_nnet_cpu.sh --num-valid-utts 150 \
   --minibatches-per-phase-it1 100 \
   --minibatches-per-phase 400 \
   --num-iters 15 --num_hidden_layers 3 \
   --num-parameters 6000000 --samples_per_iteration 800000 \
   --cmd "$decode_cmd" --parallel-opts "-pe smp 16" \
  data/train_30k_nodup data/lang exp/tri4b exp/tri5b2_nnet





