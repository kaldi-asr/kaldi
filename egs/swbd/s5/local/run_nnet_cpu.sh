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
 # optimize over.  Limiting to 10 iterations.

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