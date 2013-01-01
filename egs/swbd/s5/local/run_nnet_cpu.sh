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
  steps/decode_fmllr.sh --nj 30 --cmd "$decode_cmd" --config conf/decode.config \
   exp/tri4b/graph data/eval2000 exp/tri4b/decode_eval2000
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
## Caution--  there still seems to be instability here, it got worse at first.
(
 steps/train_nnet_cpu.sh --num_hidden_layers 3 \
   --num-iters 15 --num-parameters 6000000 --samples_per_iteration 800000 \
   --minibatch-size 250 --minibatches-per-phase-it1 200 --minibatches-per-phase 800 \
   --cmd "$decode_cmd" --parallel-opts "-pe smp 16" \
   data/train_30k_nodup data/lang exp/tri4b exp/tri5b1b_nnet

 steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 30 \
   --config conf/decode.config --transform-dir exp/tri4b/decode_train_dev \
   exp/tri4b/graph data/train_dev exp/tri5b1b_nnet/decode_train_dev

 steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 30 \
   --config conf/decode.config --transform-dir exp/tri4b/decode_eval2000 \
   exp/tri4b/graph data/train_dev exp/tri5b1b_nnet/decode_eval2000
)&
  ## evaluated likelihood of the previous on train_dev, just like Karel did, but
  ## values were similar to his.
  ##     steps/eval_nnet_like.sh --iter 15 data/train_dev/ exp/tri4b_ali_train_dev/ exp/tri5b1_nnet/

## tri5b9 will be as 5b7 (the 1st 5 iters) but just rerunning after
## changing the code to remove l2 regularization.
( 
  steps/train_nnet_cpu.sh --measure-gradient-at 0.8 --minibatch-size 250 \
  --num-valid-frames 15000 \
  --minibatches-per-phase-it1 200 --minibatches-per-phase 800 \
   --num-parameters 2000000 --samples_per_iteration 800000 \
   --cmd "$decode_cmd" --parallel-opts "-pe smp 16" \
   data/train_30k_nodup data/lang exp/tri4b exp/tri5b9_nnet

 steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 30 \
   --config conf/decode.config --transform-dir exp/tri4b/decode_train_dev \
  exp/tri4b/graph data/train_dev exp/tri5b9_nnet/decode_train_dev
)


## tri5b10 is as tri5b9, but adding --precondition true.
( 
  steps/train_nnet_cpu.sh --measure-gradient-at 0.8 --minibatch-size 250 \
  --nnet-config-opts "--precondition true" \
   --num-valid-frames 15000 \
  --minibatches-per-phase-it1 200 --minibatches-per-phase 800 \
   --num-parameters 2000000 --samples_per_iteration 800000 \
   --cmd "$decode_cmd" --parallel-opts "-pe smp 16" \
   data/train_30k_nodup data/lang exp/tri4b exp/tri5b10_nnet

 steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 30 \
   --config conf/decode.config --transform-dir exp/tri4b/decode_train_dev \
  exp/tri4b/graph data/train_dev exp/tri5b10_nnet/decode_train_dev
)


(  # Doing more iters of training tri5b10.
  steps/train_nnet_cpu.sh --measure-gradient-at 0.8 --minibatch-size 250 \
   --num-iters 15 --stage 5 \
  --nnet-config-opts "--precondition true" \
   --num-valid-frames 15000 \
  --minibatches-per-phase-it1 200 --minibatches-per-phase 800 \
   --num-parameters 2000000 --samples_per_iteration 800000 \
   --cmd "$decode_cmd" --parallel-opts "-pe smp 16" \
   data/train_30k_nodup data/lang exp/tri4b exp/tri5b10_nnet

 steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 30 --iter 15 \
   --config conf/decode.config --transform-dir exp/tri4b/decode_train_dev \
  exp/tri4b/graph data/train_dev exp/tri5b10_nnet/decode_train_dev_it15
)

(  # Rerunning 5b10 after rearranging the code so that the
  # "precondition" stuff is now called "nobias".
  steps/train_nnet_cpu.sh --measure-gradient-at 0.8 --minibatch-size 250 \
   --num-iters 15 \
  --nnet-config-opts "--nobias" \
   --num-valid-frames 15000 \
  --minibatches-per-phase-it1 200 --minibatches-per-phase 800 \
   --num-parameters 2000000 --samples_per_iteration 800000 \
   --cmd "$decode_cmd" --parallel-opts "-pe smp 16" \
   data/train_30k_nodup data/lang exp/tri4b exp/tri5b10rerun_nnet

 steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 30 --iter 15 \
   --config conf/decode.config --transform-dir exp/tri4b/decode_train_dev \
  exp/tri4b/graph data/train_dev exp/tri5b10rerun_nnet/decode_train_dev_it15
)

(  # Rerunning 5b9 after code changes.
  steps/train_nnet_cpu.sh --measure-gradient-at 0.8 --minibatch-size 250 \
   --num-iters 15 --num-valid-frames 15000 \
  --minibatches-per-phase-it1 200 --minibatches-per-phase 800 \
   --num-parameters 2000000 --samples_per_iteration 800000 \
   --cmd "$decode_cmd" --parallel-opts "-pe smp 16" \
   data/train_30k_nodup data/lang exp/tri4b exp/tri5b9rerun_nnet

 steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 30 --iter 15 \
   --config conf/decode.config --transform-dir exp/tri4b/decode_train_dev \
  exp/tri4b/graph data/train_dev exp/tri5b9rerun_nnet/decode_train_dev_it15
)


## tri5b11 is as tri5b10, but changing learning-rate-ratio from 1.1 (the
## default to 1.02.  This means the learning rate can only change very slowly,
## so it stays high for longer.   Doing a large number of iterations (20).
( 
  steps/train_nnet_cpu.sh --measure-gradient-at 0.8 --minibatch-size 250 \
  --num-iters 20 --learning-rate-ratio 1.02 \
  --nnet-config-opts "--precondition true" \
   --num-valid-frames 15000 \
  --minibatches-per-phase-it1 200 --minibatches-per-phase 800 \
   --num-parameters 2000000 --samples_per_iteration 800000 \
   --cmd "$decode_cmd" --parallel-opts "-pe smp 16" \
   data/train_30k_nodup data/lang exp/tri4b exp/tri5b11_nnet

 steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 30 \
   --config conf/decode.config --transform-dir exp/tri4b/decode_train_dev \
  exp/tri4b/graph data/train_dev exp/tri5b11_nnet/decode_train_dev
)

## tri5b12 is as tri5b11 but with an even smaller ratio (1.01) on the
## learning rate, and more iterations (30).
( 
  steps/train_nnet_cpu.sh --measure-gradient-at 0.8 --minibatch-size 250 \
  --num-iters 30 --learning-rate-ratio 1.01 \
  --nnet-config-opts "--precondition true" \
   --num-valid-frames 15000 \
  --minibatches-per-phase-it1 200 --minibatches-per-phase 800 \
   --num-parameters 2000000 --samples_per_iteration 800000 \
   --cmd "$decode_cmd" --parallel-opts "-pe smp 16" \
   data/train_30k_nodup data/lang exp/tri4b exp/tri5b12_nnet

 steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 30 \
   --config conf/decode.config --transform-dir exp/tri4b/decode_train_dev \
  exp/tri4b/graph data/train_dev exp/tri5b12_nnet/decode_train_dev
)

# tri5b13 is as tri5b10, but with alpha=1.0 (the preconditioned update),
# and using a larger minibatch size.

steps/train_nnet_cpu.sh --measure-gradient-at 0.8 --minibatch-size 1000 \
  --alpha 1.0 --num-iters 15  --num-valid-frames 15000 \
  --minibatches-per-phase-it1 50 --minibatches-per-phase 200 \
   --num-parameters 2000000 --samples_per_iteration 800000 \
   --cmd "$decode_cmd" --parallel-opts "-pe smp 16" \
   data/train_30k_nodup data/lang exp/tri4b exp/tri5b13_nnet

# tri5b14 is as tri5b13, but with a larger learning rate (doubled)
# and using a larger minibatch size.
steps/train_nnet_cpu.sh --measure-gradient-at 0.8 --minibatch-size 1000 \
  --nnet-config-opts "--learning-rate 0.002" \
  --alpha 1.0 --num-iters 15  --num-valid-frames 15000 \
  --minibatches-per-phase-it1 50 --minibatches-per-phase 200 \
   --num-parameters 2000000 --samples_per_iteration 800000 \
   --cmd "$decode_cmd" --parallel-opts "-pe smp 16" \
   data/train_30k_nodup data/lang exp/tri4b exp/tri5b14_nnet

# tri5b15 is as tri5b14, but with a learning-rate-ratio closer to 1.
steps/train_nnet_cpu.sh --measure-gradient-at 0.8 --minibatch-size 1000 \
 --learning-rate-ratio 1.02 \
  --nnet-config-opts "--learning-rate 0.002" \
  --alpha 1.0 --num-iters 15  --num-valid-frames 15000 \
  --minibatches-per-phase-it1 50 --minibatches-per-phase 200 \
   --num-parameters 2000000 --samples_per_iteration 800000 \
   --cmd "$decode_cmd" --parallel-opts "-pe smp 16" \
   data/train_30k_nodup data/lang exp/tri4b exp/tri5b15_nnet


# tri5b16 is as tri5b14, but with alpha=0.5.
steps/train_nnet_cpu.sh --measure-gradient-at 0.8 --minibatch-size 1000 \
  --nnet-config-opts "--learning-rate 0.002" \
  --alpha 0.5 --num-iters 15  --num-valid-frames 15000 \
  --minibatches-per-phase-it1 50 --minibatches-per-phase 200 \
   --num-parameters 2000000 --samples_per_iteration 800000 \
   --cmd "$decode_cmd" --parallel-opts "-pe smp 16" \
   data/train_30k_nodup data/lang exp/tri4b exp/tri5b16_nnet

# tri5b17 is as tri5b14, but with alpha=2.0 [also c.f. 16].
steps/train_nnet_cpu.sh --measure-gradient-at 0.8 --minibatch-size 1000 \
  --nnet-config-opts "--learning-rate 0.002" \
  --alpha 2.0 --num-iters 15  --num-valid-frames 15000 \
  --minibatches-per-phase-it1 50 --minibatches-per-phase 200 \
   --num-parameters 2000000 --samples_per_iteration 800000 \
   --cmd "$decode_cmd" --parallel-opts "-pe smp 16" \
   data/train_30k_nodup data/lang exp/tri4b exp/tri5b17_nnet

# tri5b18 is as tri5b14, but with alpha=4.0 [also c.f. 16,17].
steps/train_nnet_cpu.sh --measure-gradient-at 0.8 --minibatch-size 1000 \
  --nnet-config-opts "--learning-rate 0.002" \
  --alpha 4.0 --num-iters 15  --num-valid-frames 15000 \
  --minibatches-per-phase-it1 50 --minibatches-per-phase 200 \
   --num-parameters 2000000 --samples_per_iteration 800000 \
   --cmd "$decode_cmd" --parallel-opts "-pe smp 16" \
   data/train_30k_nodup data/lang exp/tri4b exp/tri5b18_nnet

 steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 30 --iter 15 \
   --config conf/decode.config --transform-dir exp/tri4b/decode_train_dev \
  exp/tri4b/graph data/train_dev exp/tri5b18_nnet/decode_train_dev

# tri5b19 is as tri5b14, but with alpha=8.0 [also c.f. 16,17,18].
steps/train_nnet_cpu.sh --measure-gradient-at 0.8 --minibatch-size 1000 \
  --nnet-config-opts "--learning-rate 0.002" \
  --alpha 8.0 --num-iters 15  --num-valid-frames 15000 \
  --minibatches-per-phase-it1 50 --minibatches-per-phase 200 \
   --num-parameters 2000000 --samples_per_iteration 800000 \
   --cmd "$decode_cmd" --parallel-opts "-pe smp 16" \
   data/train_30k_nodup data/lang exp/tri4b exp/tri5b19_nnet

 steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 30 --iter 15 \
   --config conf/decode.config --transform-dir exp/tri4b/decode_train_dev \
  exp/tri4b/graph data/train_dev exp/tri5b19_nnet/decode_train_dev

# tri5b20 is as tri5b18, but borrowing from tri5b15 the idea of using
# a learning-rate-ratio closer to 1.
steps/train_nnet_cpu.sh --measure-gradient-at 0.8 --minibatch-size 1000 \
  --learning-rate-ratio 1.02 \
  --nnet-config-opts "--learning-rate 0.002" \
  --alpha 4.0 --num-iters 15  --num-valid-frames 15000 \
  --minibatches-per-phase-it1 50 --minibatches-per-phase 200 \
   --num-parameters 2000000 --samples_per_iteration 800000 \
   --cmd "$decode_cmd" --parallel-opts "-pe smp 16" \
   data/train_30k_nodup data/lang exp/tri4b exp/tri5b20_nnet

# tri5b21 is as tri5b18, but with "shrinkage"
steps/train_nnet_cpu.sh --measure-gradient-at 0.8 --minibatch-size 1000 \
  --shrink true \
  --nnet-config-opts "--learning-rate 0.002" \
  --alpha 4.0 --num-iters 15  --num-valid-frames 15000 \
  --minibatches-per-phase-it1 50 --minibatches-per-phase 200 \
   --num-parameters 2000000 --samples_per_iteration 800000 \
   --cmd "$decode_cmd" --parallel-opts "-pe smp 16" \
   data/train_30k_nodup data/lang exp/tri4b exp/tri5b21_nnet

 steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 30 --iter 15 \
   --config conf/decode.config --transform-dir exp/tri4b/decode_train_dev \
  exp/tri4b/graph data/train_dev exp/tri5b21_nnet/decode_train_dev

# tri5b22 is as tri5b21 but with alpha=1.0; possibly a more aggressive
# update will combine well with shrinkage. [canceled this before finished.]
steps/train_nnet_cpu.sh --measure-gradient-at 0.8 --minibatch-size 1000 \
  --shrink true \
  --nnet-config-opts "--learning-rate 0.002" \
  --alpha 1.0 --num-iters 15  --num-valid-frames 15000 \
  --minibatches-per-phase-it1 50 --minibatches-per-phase 200 \
   --num-parameters 2000000 --samples_per_iteration 800000 \
   --cmd "$decode_cmd" --parallel-opts "-pe smp 16" \
   data/train_30k_nodup data/lang exp/tri4b exp/tri5b22_nnet

# tri5b23 is as tri5b21 but with 3 not 2 hidden layers.
steps/train_nnet_cpu.sh --measure-gradient-at 0.8 --minibatch-size 1000 \
  --num-hidden-layers 3 --shrink true \
  --nnet-config-opts "--learning-rate 0.002" \
  --alpha 4.0 --num-iters 15  --num-valid-frames 15000 \
  --minibatches-per-phase-it1 50 --minibatches-per-phase 200 \
   --num-parameters 2000000 --samples_per_iteration 800000 \
   --cmd "$decode_cmd" --parallel-opts "-pe smp 16" \
   data/train_30k_nodup data/lang exp/tri4b exp/tri5b23_nnet

# tri5b24 is as tri5b21 but with double the learning rate, initially.
steps/train_nnet_cpu.sh --measure-gradient-at 0.8 --minibatch-size 1000 \
  --shrink true \
  --nnet-config-opts "--learning-rate 0.004" \
  --alpha 4.0 --num-iters 15  --num-valid-frames 15000 \
  --minibatches-per-phase-it1 50 --minibatches-per-phase 200 \
   --num-parameters 2000000 --samples_per_iteration 800000 \
   --cmd "$decode_cmd" --parallel-opts "-pe smp 16" \
   data/train_30k_nodup data/lang exp/tri4b exp/tri5b24_nnet


# tri5b25 is as tri5b23 but with but with double the learning rate, initially
# [ or like tri5b24 but with 3 hidden layers]
steps/train_nnet_cpu.sh --measure-gradient-at 0.8 --minibatch-size 1000 \
  --shrink true --num-hidden-layers 3 \
  --nnet-config-opts "--learning-rate 0.004" \
  --alpha 4.0 --num-iters 15  --num-valid-frames 15000 \
  --minibatches-per-phase-it1 50 --minibatches-per-phase 200 \
   --num-parameters 2000000 --samples_per_iteration 800000 \
   --cmd "$decode_cmd" --parallel-opts "-pe smp 16" \
   data/train_30k_nodup data/lang exp/tri4b exp/tri5b25_nnet

 steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 30 --iter 15 \
   --config conf/decode.config --transform-dir exp/tri4b/decode_train_dev \
  exp/tri4b/graph data/train_dev exp/tri5b25_nnet/decode_train_dev


# tri5b26 is like tri5b25 but double the #parameters.
# [note: about 1/2 percent better.]
steps/train_nnet_cpu.sh --measure-gradient-at 0.8 --minibatch-size 1000 \
  --shrink true --num-hidden-layers 3 \
  --nnet-config-opts "--learning-rate 0.004" \
  --alpha 4.0 --num-iters 15  --num-valid-frames 15000 \
  --minibatches-per-phase-it1 50 --minibatches-per-phase 200 \
   --num-parameters 4000000 --samples_per_iteration 800000 \
   --cmd "$decode_cmd" --parallel-opts "-pe smp 16" \
   data/train_30k_nodup data/lang exp/tri4b exp/tri5b26_nnet

 steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 30 --iter 15 \
   --config conf/decode.config --transform-dir exp/tri4b/decode_train_dev \
  exp/tri4b/graph data/train_dev exp/tri5b26_nnet/decode_train_dev

# tri5b27 is as tri5b25 but with learning-rate-ratio closer to 1.
steps/train_nnet_cpu.sh --measure-gradient-at 0.8 --minibatch-size 1000 \
  --learning-rate-ratio 1.02 \
  --shrink true --num-hidden-layers 3 \
  --nnet-config-opts "--learning-rate 0.004" \
  --alpha 4.0 --num-iters 15  --num-valid-frames 15000 \
  --minibatches-per-phase-it1 50 --minibatches-per-phase 200 \
   --num-parameters 2000000 --samples_per_iteration 800000 \
   --cmd "$decode_cmd" --parallel-opts "-pe smp 16" \
   data/train_30k_nodup data/lang exp/tri4b exp/tri5b27_nnet

 steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 30 --iter 15 \
   --config conf/decode.config --transform-dir exp/tri4b/decode_train_dev \
  exp/tri4b/graph data/train_dev exp/tri5b27_nnet/decode_train_dev

# tri5b28 is like tri5b25 but the script is changed so that now it does
# LDA.
steps/train_nnet_cpu.sh --measure-gradient-at 0.8 --minibatch-size 1000 \
  --shrink true --num-hidden-layers 3 \
  --nnet-config-opts "--learning-rate 0.004" \
  --alpha 4.0 --num-iters 15  --num-valid-frames 15000 \
  --minibatches-per-phase-it1 50 --minibatches-per-phase 200 \
   --num-parameters 2000000 --samples_per_iteration 800000 \
   --cmd "$decode_cmd" --parallel-opts "-pe smp 16" \
   data/train_30k_nodup data/lang exp/tri4b exp/tri5b28_nnet

 steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 30 --iter 15 \
   --config conf/decode.config --transform-dir exp/tri4b/decode_train_dev \
  exp/tri4b/graph data/train_dev exp/tri5b28_nnet/decode_train_dev


(
 # tri5b29 is like tri5b28 (i.e. with LDA), but now with parallel training.
 # note: we're now running 8 jobs and it's with twice the data per iteration
 # we had previously (but divided among 8 jobs, as sample-per-iteration is 200k,
 # which times 8 is 16, vs. 800k in the jobs above).
 # Reducing the number of validation frames as we now see that data more (in
 # the nnet-combine stage).

 steps/train_nnet_cpu_parallel.sh --minibatch-size 1000 \
   --num-jobs-nnet 8 --shrink true --num-hidden-layers 3 \
   --nnet-config-opts "--learning-rate 0.004" \
   --alpha 4.0 --num-iters 15  --num-valid-frames 8000 \
   --num-parameters 2000000 \
   --cmd "$decode_cmd" --parallel-opts "-pe smp 8" \
   data/train_30k_nodup data/lang exp/tri4b exp/tri5b29_nnet

 for iter in 5 10 15; do
  steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 30 --iter $iter \
     --config conf/decode.config --transform-dir exp/tri4b/decode_train_dev \
    exp/tri4b/graph data/train_dev exp/tri5b29_nnet/decode_train_dev_it$iter &
 done
)

(
 # tri5b30 is like tri5b29 (with LDA + parallel training), but now without automatic
 # updating of learning rates but instead including the last iter in the space we
 # optimize over.  [note: broken, was not using that last iter.]
 # Limiting to 10 iterations.

 steps/train_nnet_cpu_parallel2.sh --minibatch-size 1000 \
   --num-jobs-nnet 8 --shrink true --num-hidden-layers 3 \
   --nnet-config-opts "--learning-rate 0.004" \
   --alpha 4.0 --num-iters 10  --num-valid-frames 8000 \
   --num-parameters 2000000 \
   --cmd "$decode_cmd" --parallel-opts "-pe smp 8" \
   data/train_30k_nodup data/lang exp/tri4b exp/tri5b30_nnet

 for iter in 5 10; do
  steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 30 --iter $iter \
     --config conf/decode.config --transform-dir exp/tri4b/decode_train_dev \
    exp/tri4b/graph data/train_dev exp/tri5b30_nnet/decode_train_dev_it$iter &
 done
)

(# tri5b31 is rerunning tri5b30 after script fix so that it uses the last iter
 # when optimizing weights.
 # tri5b31 is like tri5b29 (with LDA + parallel training), but now without automatic
 # updating of learning rates but instead including the last iter in the space we
 # optimize over. 

 steps/train_nnet_cpu_parallel2.sh --minibatch-size 1000 \
   --num-jobs-nnet 8 --shrink true --num-hidden-layers 3 \
   --nnet-config-opts "--learning-rate 0.004" \
   --alpha 4.0 --num-iters 20  --num-valid-frames 8000 \
   --num-parameters 2000000 \
   --cmd "$decode_cmd" --parallel-opts "-pe smp 8" \
   data/train_30k_nodup data/lang exp/tri4b exp/tri5b31_nnet

 for iter in 5 10 15 20; do
  steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 30 --iter $iter \
     --config conf/decode.config --transform-dir exp/tri4b/decode_train_dev \
    exp/tri4b/graph data/train_dev exp/tri5b31_nnet/decode_train_dev_it$iter &
 done
)


(
 # tri5b32 is like tri5b29 (parallel training + automatic updating of learning
 # rates), but using double the #samples per iteration, and reducing the #valid frames,
 # in order to try to balance the time taken in SGD vs. validation-set tuning.

 steps/train_nnet_cpu_parallel.sh --minibatch-size 1000 \
   --samples-per-iteration 400000 \
   --num-jobs-nnet 8 --shrink true --num-hidden-layers 3 \
   --nnet-config-opts "--learning-rate 0.004" \
   --alpha 4.0 --num-iters 15  --num-valid-frames 5000 \
   --num-parameters 2000000 \
   --cmd "$decode_cmd" --parallel-opts "-pe smp 8" \
   data/train_30k_nodup data/lang exp/tri4b exp/tri5b32_nnet

 for iter in 5 10 15; do
  steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 30 --iter $iter \
     --config conf/decode.config --transform-dir exp/tri4b/decode_train_dev \
    exp/tri4b/graph data/train_dev exp/tri5b32_nnet/decode_train_dev_it$iter &
 done
)

(# tri5b33 is like tri5b31 (no automatic updating of weights), but
 # with 400k samples per iteration like tri5b32, and with the "3" script
 # that does the randomization in parallel.

 steps/train_nnet_cpu_parallel3.sh --minibatch-size 1000 \
   --samples-per-iteration 400000 \
   --num-jobs-nnet 8 --shrink true --num-hidden-layers 3 \
   --nnet-config-opts "--learning-rate 0.004" \
   --alpha 4.0 --num-iters 20  --num-valid-frames 8000 \
   --num-parameters 2000000 \
   --cmd "$decode_cmd" --parallel-opts "-pe smp 8" \
   data/train_30k_nodup data/lang exp/tri4b exp/tri5b33_nnet

 for iter in 5 10 15 20; do
  steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 30 --iter $iter \
     --config conf/decode.config --transform-dir exp/tri4b/decode_train_dev \
    exp/tri4b/graph data/train_dev exp/tri5b33_nnet/decode_train_dev_it$iter &
 done
)


(
 # tri5b34 is as tri5b35 but 4 million not 2 million parameters;
 # continuing training to 20 iterations.

 steps/train_nnet_cpu_parallel4.sh --minibatch-size 1000 \
   --samples-per-iteration 400000 \
   --num-jobs-nnet 8 --shrink true --num-hidden-layers 3 \
   --nnet-config-opts "--learning-rate 0.004" \
   --alpha 4.0 --num-iters 20  --num-valid-frames 5000 \
   --num-parameters 4000000 \
   --cmd "$decode_cmd" --parallel-opts "-pe smp 8" \
   data/train_30k_nodup data/lang exp/tri4b exp/tri5b34_nnet

 for iter in 5 10 15 20; do
  steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 30 --iter $iter \
     --config conf/decode.config --transform-dir exp/tri4b/decode_train_dev \
    exp/tri4b/graph data/train_dev exp/tri5b34_nnet/decode_train_dev_it$iter &
 done
)


(
 # tri5b35 is like tri5b32 (400k samples per job, and automatic updating of
 # learning rates and less valid frames), but using the "4" script which
 # is faster by doing the randomization in parallel and using fewer BFGS iters.

 steps/train_nnet_cpu_parallel4.sh --minibatch-size 1000 \
   --samples-per-iteration 400000 \
   --num-jobs-nnet 8 --shrink true --num-hidden-layers 3 \
   --nnet-config-opts "--learning-rate 0.004" \
   --alpha 4.0 --num-iters 20  --num-valid-frames 5000 \
   --num-parameters 2000000 \
   --cmd "$decode_cmd" --parallel-opts "-pe smp 8" \
   data/train_30k_nodup data/lang exp/tri4b exp/tri5b35_nnet

 for iter in 5 10 15 20; do
  steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 30 --iter $iter \
     --config conf/decode.config --transform-dir exp/tri4b/decode_train_dev \
    exp/tri4b/graph data/train_dev exp/tri5b35_nnet/decode_train_dev_it$iter &
 done
)

(
 # tri5b36 is as tri5b35, but using the
 # "5" version of the script with simpler optimization and overshooting.
 # [Terminated this as it did not seem to be working, and will use default
 # samples-per-iteration and smaller --valid-impr-thresh].

 steps/train_nnet_cpu_parallel5.sh --minibatch-size 1000 \
   --samples-per-iteration 400000 \
   --num-jobs-nnet 8 --num-hidden-layers 3 \
   --nnet-config-opts "--learning-rate 0.004" \
   --alpha 4.0 --num-iters 20  --num-valid-frames 5000 \
   --num-parameters 2000000 \
   --cmd "$decode_cmd" --parallel-opts "-pe smp 8" \
   data/train_30k_nodup data/lang exp/tri4b exp/tri5b36_nnet

 for iter in 5 10 15 20; do
  steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 30 --iter $iter \
     --config conf/decode.config --transform-dir exp/tri4b/decode_train_dev \
    exp/tri4b/graph data/train_dev exp/tri5b36_nnet/decode_train_dev_it$iter &
 done
)


(
 # tri5b37 is as tri5b36, (using the
 # "5" version of the script with simpler optimization and overshooting),
 # but double the #parameters (similar to tri5b34).

 steps/train_nnet_cpu_parallel5.sh --minibatch-size 1000 \
   --samples-per-iteration 400000 \
   --num-jobs-nnet 8 --num-hidden-layers 3 \
   --nnet-config-opts "--learning-rate 0.004" \
   --alpha 4.0 --num-iters 20  --num-valid-frames 5000 \
   --num-parameters 4000000 \
   --cmd "$decode_cmd" --parallel-opts "-pe smp 8" \
   data/train_30k_nodup data/lang exp/tri4b exp/tri5b37_nnet

 for iter in 5 10 15 20; do
  steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 30 --iter $iter \
     --config conf/decode.config --transform-dir exp/tri4b/decode_train_dev \
    exp/tri4b/graph data/train_dev exp/tri5b37_nnet/decode_train_dev_it$iter &
 done
)


(
 # tri5b38 is as tri5b36 ("5" script), but back to 200k samples
 # per iteration (the default), and setting valid-impr-thresh to 0.1.


 steps/train_nnet_cpu_parallel5.sh --minibatch-size 1000 \
   --valid-impr-thresh 0.1 \
   --num-jobs-nnet 8 --num-hidden-layers 3 \
   --nnet-config-opts "--learning-rate 0.004" \
   --alpha 4.0 --num-iters 20  --num-valid-frames 5000 \
   --num-parameters 2000000 \
   --cmd "$decode_cmd" --parallel-opts "-pe smp 8" \
   data/train_30k_nodup data/lang exp/tri4b exp/tri5b38_nnet

 for iter in 5 10 15 20; do
  steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 30 --iter $iter \
     --config conf/decode.config --transform-dir exp/tri4b/decode_train_dev \
    exp/tri4b/graph data/train_dev exp/tri5b38_nnet/decode_train_dev_it$iter &
 done
)


(
 # tri5b39 is as tri5b38 ("5" script), but using twice #iters as I should
 # have last time, and also setting max-learning-rate-factor to 1.0, to
 # prevent learning rate increases.  [note: better than 5b38 even on the
 # iters that were the same, so perhaps --max-learning-rate-factor 1.0 helps.

 steps/train_nnet_cpu_parallel5.sh --minibatch-size 1000 \
   --max-learning-rate-factor 1.0 \
   --valid-impr-thresh 0.1 \
   --num-jobs-nnet 8 --num-hidden-layers 3 \
   --nnet-config-opts "--learning-rate 0.004" \
   --alpha 4.0 --num-iters 40  --num-valid-frames 5000 \
   --num-parameters 2000000 \
   --cmd "$decode_cmd" --parallel-opts "-pe smp 8" \
   data/train_30k_nodup data/lang exp/tri4b exp/tri5b39_nnet

 for iter in 10 20 30 40; do
  steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 30 --iter $iter \
     --config conf/decode.config --transform-dir exp/tri4b/decode_train_dev \
    exp/tri4b/graph data/train_dev exp/tri5b39_nnet/decode_train_dev_it$iter &
 done
)


(
 # tri5b40 is as tri5b39 ("5" scripts and --max-learning-rate-factor 1.0),
 # but adding --overshoot 1.0.
 # This is clearly worse than tri5b39, by about 2% absolute at the end.

 steps/train_nnet_cpu_parallel5.sh --minibatch-size 1000 \
   --max-learning-rate-factor 1.0 --overshoot 1.0 \
   --valid-impr-thresh 0.1 \
   --num-jobs-nnet 8 --num-hidden-layers 3 \
   --nnet-config-opts "--learning-rate 0.004" \
   --alpha 4.0 --num-iters 40  --num-valid-frames 5000 \
   --num-parameters 2000000 \
   --cmd "$decode_cmd" --parallel-opts "-pe smp 8" \
   data/train_30k_nodup data/lang exp/tri4b exp/tri5b40_nnet

 for iter in 10 20 30 40; do
  steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 30 --iter $iter \
     --config conf/decode.config --transform-dir exp/tri4b/decode_train_dev \
    exp/tri4b/graph data/train_dev exp/tri5b40_nnet/decode_train_dev_it$iter &
 done
)


(
 # tri5b41 is using the "parallel6.sh" script, which updates learning
 # rates according to  fixed step size.    Using the same learning rate
 # schedule as in the RM setup (should play with this), and 40 iters.
 # CAUTION-- this is an error, the final-learning-rate is 10 x too high.
 # but we'll see what happens.

 steps/train_nnet_cpu_parallel6.sh --minibatch-size 1000 \
   --initial-learning-rate 0.0025 --final-learning-rate 0.0025 \
   --num-jobs-nnet 8 --num-hidden-layers 3 \
   --alpha 4.0 --num-iters 40 \
   --num-parameters 2000000 \
   --cmd "$decode_cmd" --parallel-opts "-pe smp 8" \
   data/train_30k_nodup data/lang exp/tri4b exp/tri5b41_nnet

  steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 30 \
    --config conf/decode.config --transform-dir exp/tri4b/decode_train_dev \
   exp/tri4b/graph data/train_dev exp/tri5b41_nnet/decode_train_dev &
)


(
 # tri5b42 is as tri5b41 (using the "parallel6.sh" script), but larger
 # learning rate (start at 0.004), similar to what we've been using
 # in this setup previously.

 steps/train_nnet_cpu_parallel6.sh --minibatch-size 1000 \
   --initial-learning-rate 0.004 --final-learning-rate 0.0004 \
   --num-jobs-nnet 8 --num-hidden-layers 3 \
   --alpha 4.0 --num-iters 40 \
   --num-parameters 2000000 \
   --cmd "$decode_cmd" --parallel-opts "-pe smp 8" \
   data/train_30k_nodup data/lang exp/tri4b exp/tri5b42_nnet

  steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 30 \
    --config conf/decode.config --transform-dir exp/tri4b/decode_train_dev \
   exp/tri4b/graph data/train_dev exp/tri5b42_nnet/decode_train_dev &
)

(
 # tri5b43 is as tri5b42 but using smaller final learning rate.

 steps/train_nnet_cpu_parallel6.sh --minibatch-size 1000 \
   --initial-learning-rate 0.004 --final-learning-rate 0.0002 \
   --num-jobs-nnet 8 --num-hidden-layers 3 \
   --alpha 4.0 --num-iters 40 \
   --num-parameters 2000000 \
   --cmd "$decode_cmd" --parallel-opts "-pe smp 8" \
   data/train_30k_nodup data/lang exp/tri4b exp/tri5b43_nnet

  steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 30 \
    --config conf/decode.config --transform-dir exp/tri4b/decode_train_dev \
   exp/tri4b/graph data/train_dev exp/tri5b43_nnet/decode_train_dev &
)

(
 # tri5b44 is as tri5b43 but more iters (80 vs 40).

 steps/train_nnet_cpu_parallel6.sh --minibatch-size 1000 \
   --initial-learning-rate 0.004 --final-learning-rate 0.0002 \
   --num-jobs-nnet 8 --num-hidden-layers 3 \
   --alpha 4.0 --num-iters 80 \
   --num-parameters 2000000 \
   --cmd "$decode_cmd" --parallel-opts "-pe smp 8" \
   data/train_30k_nodup data/lang exp/tri4b exp/tri5b44_nnet

  steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 30 \
    --config conf/decode.config --transform-dir exp/tri4b/decode_train_dev \
   exp/tri4b/graph data/train_dev exp/tri5b44_nnet/decode_train_dev &
)



 ( # tri5b45 is as tri5b43 but larger initial learning rate.
   # Note: training objf, test objf and WER are all worse.
   steps/train_nnet_cpu_parallel6.sh --minibatch-size 1000 \
    --initial-learning-rate 0.008 --final-learning-rate 0.0004 \
    --num-jobs-nnet 8 --num-hidden-layers 3 \
    --alpha 4.0 --num-iters 40 \
     --num-parameters 2000000 \
     --cmd "$decode_cmd" --parallel-opts "-pe smp 8" \
     data/train_30k_nodup data/lang exp/tri4b exp/tri5b45_nnet; 
   steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 30 \
      --config conf/decode.config --transform-dir exp/tri4b/decode_train_dev \
    exp/tri4b/graph data/train_dev exp/tri5b45_nnet/decode_train_dev
 ) &

(
 # tri5b46 is as tri5b43 but using the "7" script which does
 # shrinking.
 steps/train_nnet_cpu_parallel7.sh --minibatch-size 1000 \
   --initial-learning-rate 0.004 --final-learning-rate 0.0002 \
   --num-jobs-nnet 8 --num-hidden-layers 3 \
   --alpha 4.0 --num-iters 40 \
   --num-parameters 2000000 \
   --cmd "$decode_cmd" --parallel-opts "-pe smp 8" \
   data/train_30k_nodup data/lang exp/tri4b exp/tri5b46_nnet

  steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 30 \
    --config conf/decode.config --transform-dir exp/tri4b/decode_train_dev \
   exp/tri4b/graph data/train_dev exp/tri5b46_nnet/decode_train_dev &
)


(
 # tri5b47 is as tri5b46 but double the initial learning rate;
 # note, this was worse in 43->45 but there was no shrinking there;
 # I think with shrinking the picture might be different.
 steps/train_nnet_cpu_parallel7.sh --minibatch-size 1000 \
   --initial-learning-rate 0.008 --final-learning-rate 0.0002 \
   --num-jobs-nnet 8 --num-hidden-layers 3 \
   --alpha 4.0 --num-iters 40 \
   --num-parameters 2000000 \
   --cmd "$decode_cmd" --parallel-opts "-pe smp 8" \
   data/train_30k_nodup data/lang exp/tri4b exp/tri5b47_nnet

  steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 30 \
    --config conf/decode.config --transform-dir exp/tri4b/decode_train_dev \
   exp/tri4b/graph data/train_dev exp/tri5b47_nnet/decode_train_dev &
)


(
 # tri5b48 is as tri5b47 but larger final learning rate.
 # Final validation objf was slightly better than 47, -2.08 -> -2.07,
 # but WER was a bit worse.  (Note: this was also what we
 # saw in the RM setup, although it's probably not significant in
 # either case.)
 steps/train_nnet_cpu_parallel7.sh --minibatch-size 1000 \
   --initial-learning-rate 0.008 --final-learning-rate 0.0008 \
   --num-jobs-nnet 8 --num-hidden-layers 3 \
   --alpha 4.0 --num-iters 40 \
   --num-parameters 2000000 \
   --cmd "$decode_cmd" --parallel-opts "-pe smp 8" \
   data/train_30k_nodup data/lang exp/tri4b exp/tri5b48_nnet

  steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 30 \
    --config conf/decode.config --transform-dir exp/tri4b/decode_train_dev \
   exp/tri4b/graph data/train_dev exp/tri5b48_nnet/decode_train_dev &
)

(
 # tri5b49 is as tri5b48 but using the "8" script which does 
 # "combine" on each stage instead of averaging then shrinking.
 #  This does seem to be a bit better than 48, 32.02->31.85, but probably
 # random.  The validation objf at the end is very slightly better, -2.07 -> 2.06.
 # not sure that I believe this.    Early on in training, 49 was worse.
 steps/train_nnet_cpu_parallel8.sh --minibatch-size 1000 \
   --initial-learning-rate 0.008 --final-learning-rate 0.0008 \
   --num-jobs-nnet 8 --num-hidden-layers 3 \
   --alpha 4.0 --num-iters 40 \
   --num-parameters 2000000 \
   --cmd "$decode_cmd" --parallel-opts "-pe smp 8" \
   data/train_30k_nodup data/lang exp/tri4b exp/tri5b49_nnet

  steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 30 \
    --config conf/decode.config --transform-dir exp/tri4b/decode_train_dev \
   exp/tri4b/graph data/train_dev exp/tri5b49_nnet/decode_train_dev &
)

(
 # tri5b50 is as tri5b48 but larger initial learning rate and smaller final.
 # (also c.f. 47.)  Note: doesn't seem to make much difference either way, but
 # about 0.1% better.
 steps/train_nnet_cpu_parallel7.sh --minibatch-size 1000 \
   --initial-learning-rate 0.008 --final-learning-rate 0.0004 \
   --num-jobs-nnet 8 --num-hidden-layers 3 \
   --alpha 4.0 --num-iters 40 \
   --num-parameters 2000000 \
   --cmd "$decode_cmd" --parallel-opts "-pe smp 8" \
   data/train_30k_nodup data/lang exp/tri4b exp/tri5b50_nnet

  steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 30 \
    --config conf/decode.config --transform-dir exp/tri4b/decode_train_dev \
   exp/tri4b/graph data/train_dev exp/tri5b50_nnet/decode_train_dev &
)


(
 # tri5b51 may be compared to tri5b50, but we're using larger both
 # initial and final learning rates to make it more like the RM setup (see 4y29).
 # It's still not quite comparable to that because it has more samples per iter.
 # It's more than 1% absolute worse-- clearly these learning rates are too high.
 steps/train_nnet_cpu_parallel7.sh --minibatch-size 1000 \
   --initial-learning-rate 0.02 --final-learning-rate 0.002 \
   --num-jobs-nnet 8 --num-hidden-layers 3 \
   --alpha 4.0 --num-iters 40 \
   --num-parameters 2000000 \
   --cmd "$decode_cmd" --parallel-opts "-pe smp 8" \
   data/train_30k_nodup data/lang exp/tri4b exp/tri5b51_nnet

  steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 30 \
    --config conf/decode.config --transform-dir exp/tri4b/decode_train_dev \
   exp/tri4b/graph data/train_dev exp/tri5b51_nnet/decode_train_dev &
)


( 
  # tri5b52 is as tri5b48 but using the "9" script which is 
  # faster than the "7" script.  WER is better 32.02->31.68.
  steps/train_nnet_cpu_parallel9.sh --minibatch-size 1000 \
    --initial-learning-rate 0.008 --final-learning-rate 0.0008 \
   --num-jobs-nnet 8 --num-hidden-layers 3 \
   --alpha 4.0 --num-iters 40 \
   --num-parameters 2000000 \
   --cmd "$decode_cmd" --parallel-opts "-pe smp 8" \
   data/train_30k_nodup data/lang exp/tri4b exp/tri5b52_nnet

  steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 30 \
    --config conf/decode.config --transform-dir exp/tri4b/decode_train_dev \
   exp/tri4b/graph data/train_dev exp/tri5b52_nnet/decode_train_dev &
)

( 
  # tri5b54 is as tri5b52 but using the "10" script which is more
  # stable than the "9" script.  (Does the shrinking every 3 iters
  # and in between, applies the parameters from the last time we
  # shrunk.)
  # Likelihood on validation is
  # worse (-2.067 -> -2.079) and WER is worse 32.02 -> 31.68.
  # Possibly it's due to waiting between adding layers.  I'm going
  # to run a version below that waits 1 iter (not 2) between adding layers.

  steps/train_nnet_cpu_parallel10.sh --minibatch-size 1000 \
    --initial-learning-rate 0.008 --final-learning-rate 0.0008 \
   --num-jobs-nnet 8 --num-hidden-layers 3 \
   --alpha 4.0 --num-iters 40 \
   --num-parameters 2000000 \
   --cmd "$decode_cmd" --parallel-opts "-pe smp 8" \
   data/train_30k_nodup data/lang exp/tri4b exp/tri5b54_nnet

  steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 30 \
    --config conf/decode.config --transform-dir exp/tri4b/decode_train_dev \
   exp/tri4b/graph data/train_dev exp/tri5b54_nnet/decode_train_dev &
)

( 
 # tri5b54b it as tri5b54 but using --local-balance false.
 # Almost the same: WER 32.02 -> 31.92.
 steps/train_nnet_cpu_parallel10.sh --minibatch-size 1000 \
   --local-balance false \
   --initial-learning-rate 0.008 --final-learning-rate 0.0008 \
   --num-jobs-nnet 8 --num-hidden-layers 3 \
   --alpha 4.0 --num-iters 40 \
   --num-parameters 2000000 \
   --cmd "$decode_cmd" --parallel-opts "-pe smp 8" \
   data/train_30k_nodup data/lang exp/tri4b exp/tri5b54b_nnet

  steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 30 \
    --config conf/decode.config --transform-dir exp/tri4b/decode_train_dev \
   exp/tri4b/graph data/train_dev exp/tri5b54b_nnet/decode_train_dev &
)


( 
  # tri5b55 is as tri5b54 but using double the #parameters: 4M vs 2M.
  # (Using the "10" script).
  # It's better: 30.79%, vs. 32.02%.

  steps/train_nnet_cpu_parallel10.sh --minibatch-size 1000 \
    --initial-learning-rate 0.008 --final-learning-rate 0.0008 \
   --num-jobs-nnet 8 --num-hidden-layers 3 \
   --alpha 4.0 --num-iters 40 \
   --num-parameters 4000000 \
   --cmd "$decode_cmd" --parallel-opts "-pe smp 8" \
   data/train_30k_nodup data/lang exp/tri4b exp/tri5b55_nnet

  steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 30 \
    --config conf/decode.config --transform-dir exp/tri4b/decode_train_dev \
   exp/tri4b/graph data/train_dev exp/tri5b55_nnet/decode_train_dev &
)

( 
  #  tri5b55b is as tri5b55, but using the "10b" script which
  # supports "--apply-shrinking false", so we only shrink
  # every 3 iters (after the first few iters).
  # WER non-significantly better, 30.79 -> 30.73.
  steps/train_nnet_cpu_parallel10b.sh --minibatch-size 1000 \
   --apply-shrinking false \
    --initial-learning-rate 0.008 --final-learning-rate 0.0008 \
   --num-jobs-nnet 8 --num-hidden-layers 3 \
   --alpha 4.0 --num-iters 40 \
   --num-parameters 4000000 \
   --cmd "$decode_cmd" --parallel-opts "-pe smp 8" \
   data/train_30k_nodup data/lang exp/tri4b exp/tri5b55b_nnet

  steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 30 \
    --config conf/decode.config --transform-dir exp/tri4b/decode_train_dev \
   exp/tri4b/graph data/train_dev exp/tri5b55b_nnet/decode_train_dev &
)


(
 # tri5b56 is as tri5b54 but waiting 1 not 2 iters between adding
 # layers. 
 # It's actually a bit worse: 32.02 -> 32.24% WER, and the validatin
 # performance is worse, -2.08 -> -2.09.

  steps/train_nnet_cpu_parallel10.sh --minibatch-size 1000 \
   --add-layers-period 1 --initial-learning-rate 0.008 --final-learning-rate 0.0008 \
   --num-jobs-nnet 8 --num-hidden-layers 3 \
   --alpha 4.0 --num-iters 40 \
   --num-parameters 2000000 \
   --cmd "$decode_cmd" --parallel-opts "-pe smp 8" \
   data/train_30k_nodup data/lang exp/tri4b exp/tri5b56_nnet

  steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 30 \
    --config conf/decode.config --transform-dir exp/tri4b/decode_train_dev \
   exp/tri4b/graph data/train_dev exp/tri5b56_nnet/decode_train_dev
)


(
 # tri5b57 is as tri5b54 (also c.f. tri5b56), but adding all the layers
 # at the beginning.
 # It's worse: 32.02 -> 32.31% WER, and the validatin
 # performance is worse, -2.08 -> -2.11.
 
  steps/train_nnet_cpu_parallel10.sh --minibatch-size 1000 \
   --initial-learning-rate 0.008 --final-learning-rate 0.0008 \
   --num-jobs-nnet 8 --initial-num-hidden-layers 3 --num-hidden-layers 3 \
   --alpha 4.0 --num-iters 40 \
   --num-parameters 2000000 \
   --cmd "$decode_cmd" --parallel-opts "-pe smp 8" \
   data/train_30k_nodup data/lang exp/tri4b exp/tri5b57_nnet

  steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 30 \
    --config conf/decode.config --transform-dir exp/tri4b/decode_train_dev \
   exp/tri4b/graph data/train_dev exp/tri5b57_nnet/decode_train_dev
)



( 
  # tri5b58 is as tri5b54 but using the "12" script which
  # builds a two-level tree.  This is a baseline for the next
  # experiment; we don't use the mixture components here.
  # WER almost the same, 32.02 -> 32.18.

  steps/train_nnet_cpu_parallel12.sh --minibatch-size 1000 \
   --use-mixtures false \
   --initial-learning-rate 0.008 --final-learning-rate 0.0008 \
   --num-jobs-nnet 8 --num-hidden-layers 3 \
   --alpha 4.0 --num-iters 40 \
   --num-parameters 2000000 \
   --cmd "$decode_cmd" --parallel-opts "-pe smp 8" \
   data/train_30k_nodup data/lang exp/tri4b exp/tri5b58_nnet 500 2500

  utils/mkgraph.sh data/lang_test exp/tri5b58_nnet exp/tri5b58_nnet/graph

  steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 30 \
    --config conf/decode.config --transform-dir exp/tri4b/decode_train_dev \
   exp/tri5b58_nnet/graph data/train_dev exp/tri5b58_nnet/decode_train_dev
)

( 
 # tri5b59 is as tri5b58, but using the mixture-probabilities with the
 # two-level tree.
 # Only slightly better than its baseline (58):  32.18 -> 31.97.

  steps/train_nnet_cpu_parallel12.sh --minibatch-size 1000 \
   --use-mixtures true \
   --initial-learning-rate 0.008 --final-learning-rate 0.0008 \
   --num-jobs-nnet 8 --num-hidden-layers 3 \
   --alpha 4.0 --num-iters 40 \
   --num-parameters 2000000 \
   --cmd "$decode_cmd" --parallel-opts "-pe smp 8" \
   data/train_30k_nodup data/lang exp/tri4b exp/tri5b59_nnet 500 2500

  utils/mkgraph.sh data/lang_test exp/tri5b59_nnet exp/tri5b59_nnet/graph

  steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 30 \
    --config conf/decode.config --transform-dir exp/tri4b/decode_train_dev \
   exp/tri5b59_nnet/graph data/train_dev exp/tri5b59_nnet/decode_train_dev
)



### From here is experiments on the larger data set.


( 
  # tri5c1 is similar to tri5b55, but doing it on a larger amount of
  # data and from a bigger initial SAT model (4a, 100k_nodup). 
  # Going from 3->4 hidden layers and 4->8 million parameters,
  # similar to what Shakti was using.  For now, still using
  # the same #iters (40), but double the #jobs.
  # using --local-balance false to speed up the "randomize" process
  # which was taking too long.

 steps/train_nnet_cpu_parallel10.sh --minibatch-size 1000 \
   --local-balance false \
   --initial-learning-rate 0.008 --final-learning-rate 0.0008 \
   --num-jobs-nnet 16 --num-hidden-layers 4 \
   --alpha 4.0 --num-iters 40 \
   --num-parameters 8000000 \
   --cmd "$decode_cmd" --parallel-opts "-pe smp 8" \
   data/train_100k_nodup data/lang exp/tri4a exp/tri5c1_nnet

  steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 30 \
    --config conf/decode.config --transform-dir exp/tri4a/decode_train_dev \
   exp/tri4a/graph data/train_dev exp/tri5c1_nnet/decode_train_dev &
)

( 
 # tri5c1b is as tri5c1, but more iterations (40 -> 80).

 steps/train_nnet_cpu_parallel10.sh --minibatch-size 1000 \
   --local-balance false \
   --initial-learning-rate 0.008 --final-learning-rate 0.0008 \
   --num-jobs-nnet 16 --num-hidden-layers 4 \
   --alpha 4.0 --num-iters 80 \
   --num-parameters 8000000 \
   --cmd "$decode_cmd" --parallel-opts "-pe smp 8" \
   data/train_100k_nodup data/lang exp/tri4a exp/tri5c1b_nnet

  steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 30 \
    --config conf/decode.config --transform-dir exp/tri4a/decode_train_dev \
   exp/tri4a/graph data/train_dev exp/tri5c1b_nnet/decode_train_dev &
)

( 
 # tri5c1c is as tri5c1b (80 iterations), but
 # doubling the samples per iteration, so we have
 # another factor of 2 on the amount of data seen.
 # Also, for speed, setting apply-shrinking to false,
 # so we only shrink every 3 iters.
 steps/train_nnet_cpu_parallel10b.sh --minibatch-size 1000 \
   --apply-shrinking false \
   --local-balance false \
   --samples-per-iteration 400000 \
   --initial-learning-rate 0.008 --final-learning-rate 0.0008 \
   --num-jobs-nnet 16 --num-hidden-layers 4 \
   --alpha 4.0 --num-iters 80 \
   --num-parameters 8000000 \
   --cmd "$decode_cmd" --parallel-opts "-pe smp 8" \
   data/train_100k_nodup data/lang exp/tri4a exp/tri5c1c_nnet

  steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 30 \
    --config conf/decode.config --transform-dir exp/tri4a/decode_train_dev \
   exp/tri4a/graph data/train_dev exp/tri5c1c_nnet/decode_train_dev &
)

( 
 # tri5c1d is as tri5c1c (80 iterations, double samples per iter),
 # but using the newer code [warning, using temporary Path.sh script],
 # and mixing up.  We have about 3k leaves in this system, so mixnig
 # up to 8k. 
 steps/train_nnet_cpu_parallel10c.sh --minibatch-size 1000 \
   --mix-up 8000 \
   --apply-shrinking false \
   --samples-per-iteration 400000 \
   --initial-learning-rate 0.008 --final-learning-rate 0.0008 \
   --num-jobs-nnet 16 --num-hidden-layers 4 \
   --alpha 4.0 --num-iters 80 \
   --num-parameters 8000000 \
   --cmd "$decode_cmd" --parallel-opts "-pe smp 8" \
   data/train_100k_nodup data/lang exp/tri4a exp/tri5c1d_nnet

  steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 30 \
    --config conf/decode.config --transform-dir exp/tri4a/decode_train_dev \
   exp/tri4a/graph data/train_dev exp/tri5c1d_nnet/decode_train_dev &
)


( 
 # tri5c2 is the batch update, mixing up again to 8k, as tri5c1d.
 steps/train_nnet_cpu_batch.sh --minibatch-size 1024 \
   --stage 0 --add-layers-period 1 --num-sgd-iters 6 \
   --mix-up 8000 \
   --samples-per-iteration 400000 \
   --initial-learning-rate 0.01 --final-learning-rate 0.005 \
   --num-jobs-sgd 16 --num-hidden-layers 4 \
   --alpha 4.0 \
   --num-parameters 8000000 \
   --cmd "$decode_cmd" --parallel-opts "-pe smp 8" \
   data/train_100k_nodup data/lang exp/tri4a exp/tri5c2_nnet

  steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 30 \
    --config conf/decode.config --transform-dir exp/tri4a/decode_train_dev \
   exp/tri4a/graph data/train_dev exp/tri5c2_nnet/decode_train_dev &
)


( 
 # tri5c3 is as 5c2, but using the batch2.sh script which
 # splits the data into multiple parts; I'm setting it to use
 # --data-parts=4 (1/4 the data each time), and 40 batch iters.
 cp -r exp/tri5c{2,3}_nnet # to get stuff from initial iters.
 steps/train_nnet_cpu_batch2.sh --stage 6  \
   --minibatch-size 1024 \
   --num-batch-iters 40 --data-parts 4 \
   --add-layers-period 1 --num-sgd-iters 6 \
   --mix-up 8000 \
   --samples-per-iteration 400000 \
   --initial-learning-rate 0.01 --final-learning-rate 0.005 \
   --num-jobs-sgd 16 --num-hidden-layers 4 \
   --alpha 4.0 \
   --num-parameters 8000000 \
   --cmd "$decode_cmd" --parallel-opts "-pe smp 8" \
   data/train_100k_nodup data/lang exp/tri4a exp/tri5c3_nnet

  steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 30 \
    --config conf/decode.config --transform-dir exp/tri4a/decode_train_dev \
   exp/tri4a/graph data/train_dev exp/tri5c3_nnet/decode_train_dev &
)

( 
 # tri5c4 is as tri5c3, but using the "batch3" script which mostly
 # optimizes using a subset of the training set, then switches to a
 # validation set near the end.  Worse;
 # like tri5c3, it seems to get stuck.
 steps/train_nnet_cpu_batch3.sh \
   --minibatch-size 1024 \
   --num-batch-iters 40 --data-parts 4 \
   --add-layers-period 1 --num-sgd-iters 6 \
   --mix-up 8000 \
   --samples-per-iteration 400000 \
   --initial-learning-rate 0.01 --final-learning-rate 0.005 \
   --num-jobs-sgd 16 --num-hidden-layers 4 \
   --alpha 4.0 \
   --num-parameters 8000000 \
   --cmd "$decode_cmd" --parallel-opts "-pe smp 8" \
   data/train_100k_nodup data/lang exp/tri4a exp/tri5c4_nnet

  steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 30 \
    --config conf/decode.config --transform-dir exp/tri4a/decode_train_dev \
   exp/tri4a/graph data/train_dev exp/tri5c4_nnet/decode_train_dev &
)


(
 # tri5c5 is as tri5c3, but using half-and-half training and
 # validation frames in the "validation set".

 steps/train_nnet_cpu_batch2.sh \
    --num-valid-frames 2000 --num-train-frames-in-valid 2000 \
   --minibatch-size 1024 \
   --num-batch-iters 40 --data-parts 4 \
   --add-layers-period 1 --num-sgd-iters 6 \
   --mix-up 8000 \
   --samples-per-iteration 400000 \
   --initial-learning-rate 0.01 --final-learning-rate 0.005 \
   --num-jobs-sgd 16 --num-hidden-layers 4 \
   --alpha 4.0 \
   --num-parameters 8000000 \
   --cmd "$decode_cmd" --parallel-opts "-pe smp 8" \
   data/train_100k_nodup data/lang exp/tri4a exp/tri5c5_nnet

  steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 30 \
    --config conf/decode.config --transform-dir exp/tri4a/decode_train_dev \
   exp/tri4a/graph data/train_dev exp/tri5c5_nnet/decode_train_dev &
)