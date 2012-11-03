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

  
  # This is just for a diagnostic to compare with Karel's setup-- it's 
   # not really part of the recipe.
  steps/align_fmllr.sh --nj 30 --cmd "$train_cmd" \
    data/train_dev data/lang exp/tri4b exp/tri4b_ali_train_dev



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

steps/train_nnet_cpu.sh --stage 5 --num-iters 15 --num_hidden_layers 3 \
   --num-parameters 6000000 --samples_per_iteration 800000 \
   --cmd "$decode_cmd" --parallel-opts "-pe smp 16" \
  data/train_30k_nodup data/lang exp/tri4b exp/tri5b1_nnet

steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 30 \
  --config conf/decode.config --transform-dir exp/tri4b/decode_train_dev \
  exp/tri4b/graph data/train_dev exp/tri5b1_nnet/decode_train_dev


steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 30 --iter 10 \
  --config conf/decode.config --transform-dir exp/tri4b/decode_train_dev \
  exp/tri4b/graph data/train_dev exp/tri5b1_nnet/decode_train_dev

# using more validation utts and more minibatches per phase.
steps/train_nnet_cpu.sh --num-valid-utts 150 \
   --minibatches-per-phase-it1 100 \
   --minibatches-per-phase 400 \
   --num-iters 15 --num_hidden_layers 3 \
   --num-parameters 6000000 --samples_per_iteration 800000 \
   --cmd "$decode_cmd" --parallel-opts "-pe smp 16" \
  data/train_30k_nodup data/lang exp/tri4b exp/tri5b2_nnet



steps/train_nnet_cpu.sh --num-iters 10 --num-parameters 4000000 --samples_per_iteration 800000 \
   --cmd "$decode_cmd" --parallel-opts "-pe smp 16" \
  data/train_30k_nodup data/lang exp/tri4b exp/tri5b3_nnet

steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 30 \
  --config conf/decode.config --transform-dir exp/tri4b/decode_train_dev \
  exp/tri4b/graph data/train_dev exp/tri5b3_nnet/decode_train_dev

steps/train_nnet_cpu.sh --learning-rate-ratio 1.0 \
  --num-iters 5 --num-parameters 4000000 --samples_per_iteration 800000 \
  --cmd "$decode_cmd" --parallel-opts "-pe smp 16" \
  data/train_30k_nodup data/lang exp/tri4b exp/tri5b4_nnet


# fewer samples than e.g. 5b1.  I think I also changed how the validation
# samples are selected, since then.
steps/train_nnet_cpu.sh --measure-gradient-at 0.8 \
  --num-iters 5 --num-parameters 2000000 --samples_per_iteration 800000 \
  --cmd "$decode_cmd" --parallel-opts "-pe smp 16" \
  data/train_30k_nodup data/lang exp/tri4b exp/tri5b5_nnet

steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 30 \
  --config conf/decode.config --transform-dir exp/tri4b/decode_train_dev \
  exp/tri4b/graph data/train_dev exp/tri5b5_nnet/decode_train_dev

# as 5b5, but no l2 penalty.  [Turns out that due to a bug, the l2 penalty
# was not being set to zero as it shuld
steps/train_nnet_cpu.sh --measure-gradient-at 0.8 \
  --nnet-config-opts "--l2-penalty 0.0" \
  --num-iters 5 --num-parameters 2000000 --samples_per_iteration 800000 \
  --cmd "$decode_cmd" --parallel-opts "-pe smp 16" \
  data/train_30k_nodup data/lang exp/tri4b exp/tri5b6_nnet

steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 30 \
  --config conf/decode.config --transform-dir exp/tri4b/decode_train_dev \
  exp/tri4b/graph data/train_dev exp/tri5b6_nnet/decode_train_dev


( # Running with smaller minibatch size-- starting to get desperate.
  # Otherwise similar to 5b6.  Seems to be slightly better (~0.5%).
  # A bit better cross-entopy-- 2.94 -> 2.90 after the 4th iteration.
  steps/train_nnet_cpu.sh --measure-gradient-at 0.8 --minibatch-size 250 \
  --minibatches-per-phase-it1 200 --minibatches-per-phase 800 \
   --num-iters 5 --num-parameters 2000000 --samples_per_iteration 800000 \
   --cmd "$decode_cmd" --parallel-opts "-pe smp 16" \
   data/train_30k_nodup data/lang exp/tri4b exp/tri5b7_nnet

 steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 30 \
   --config conf/decode.config --transform-dir exp/tri4b/decode_train_dev \
   exp/tri4b/graph data/train_dev exp/tri5b7_nnet/decode_train_dev
)

( # Doing the above for more iterations and re-testing.
  steps/train_nnet_cpu.sh --stage 5 --num-iters 15 --measure-gradient-at 0.8 --minibatch-size 250 \
  --minibatches-per-phase-it1 200 --minibatches-per-phase 800 \
   --num-parameters 2000000 --samples_per_iteration 800000 \
   --cmd "$decode_cmd" --parallel-opts "-pe smp 16" \
   data/train_30k_nodup data/lang exp/tri4b exp/tri5b7_nnet

 steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 30 \
   --config conf/decode.config --transform-dir exp/tri4b/decode_train_dev \
   --iter 14 exp/tri4b/graph data/train_dev exp/tri5b7_nnet/decode_train_dev_it14
)

( # as 5b7 but more parameters (4  million)
  steps/train_nnet_cpu.sh --num-iters 15 --measure-gradient-at 0.8 --minibatch-size 250 \
  --minibatches-per-phase-it1 200 --minibatches-per-phase 800 \
   --num-parameters 4000000 --samples_per_iteration 800000 \
   --cmd "$decode_cmd" --parallel-opts "-pe smp 16" \
   data/train_30k_nodup data/lang exp/tri4b exp/tri5b8_nnet

 steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 30 \
   --config conf/decode.config --transform-dir exp/tri4b/decode_train_dev \
   --iter 15 exp/tri4b/graph data/train_dev exp/tri5b8_nnet/decode_train_dev_it15
)

## Rerunning the quite-large 5b1 setup with more recent code (and with
## bias-stddev=2.0 now by default, for 15 iterations.
## Also reducing minibatch-size to 250- I think I got instability on the 1st iter.
(
 steps/train_nnet_cpu.sh --num_hidden_layers 3 \
   --num-iters 15 --num-parameters 6000000 --samples_per_iteration 800000 \
   --minibatch-size 250 --minibatches-per-phase-it1 200 --minibatches-per-phase 800 \
   --cmd "$decode_cmd" --parallel-opts "-pe smp 16" \
   data/train_30k_nodup data/lang exp/tri4b exp/tri5b1b_nnet

 steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 30 \
   --config conf/decode.config --transform-dir exp/tri4b/decode_train_dev \
   exp/tri4b/graph data/train_dev exp/tri5b1b_nnet/decode_train_dev
)&
  ## evaluated likelihood of the previous on train_dev, just like Karel did, but
  ## values were similar to his.
  ##     steps/eval_nnet_like.sh --iter 15 data/train_dev/ exp/tri4b_ali_train_dev/ exp/tri5b1_nnet/