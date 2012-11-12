#!/bin/bash

# WARNING to all those who enter here.
# This setup works OK on RM, but has not worked well on other
# setups-- it's been as much as 10% absolute worse than Karel's setup.
# We think that the issue is, the automatic setting of learning-rates is
# setting them lower than they should be.  We're working on it.
# Just be aware of the issue.

. cmd.sh

steps/train_nnet_cpu.sh data/train data/lang exp/tri3b_ali exp/tri4b_nnet

steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
  --transform-dir exp/tri3b/decode \
  exp/tri3b/graph data/test exp/tri4b_nnet/decode

for iter in 1 2 3 4; do
  steps/decode_nnet_cpu.sh --cmd "$decode_cmd -pe smp 5" --nj 20 \
    --iter $iter \
    --transform-dir exp/tri3b/decode \
   exp/tri3b/graph data/test exp/tri4b_nnet/decode_it$iter &
done


steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
  --acwt 0.2 --beam 30.0 --lat-beam 15.0 \
  --transform-dir exp/tri3b/decode \
  exp/tri3b/graph data/test exp/tri4b_nnet/decode2



steps/train_nnet_cpu.sh --cmd "queue.pl -pe smp 15" \
   data/train data/lang exp/tri3b_ali exp/tri4c_nnet

steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
  --transform-dir exp/tri3b/decode \
  exp/tri3b/graph data/test exp/tri4c_nnet/decode


for iter in 1 2 3 4; do
  steps/decode_nnet_cpu.sh --cmd "$decode_cmd -pe smp 5" --nj 20 \
    --iter $iter \
    --transform-dir exp/tri3b/decode \
   exp/tri3b/graph data/test exp/tri4c_nnet/decode_it$iter &
done

# should be like 4c, just doing it with a newer script that
# is more configurable.
steps/train_nnet_cpu.sh --cmd "queue.pl -pe smp 15" \
   data/train data/lang exp/tri3b_ali exp/tri4d_nnet

# should be same as 4c.
steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
  --transform-dir exp/tri3b/decode \
  exp/tri3b/graph data/test exp/tri4d_nnet/decode

( #1m parameters.
  steps/train_nnet_cpu.sh --cmd "queue.pl -pe smp 15" \
  --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4e_nnet

 steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
   --transform-dir exp/tri3b/decode \
   exp/tri3b/graph data/test exp/tri4e_nnet/decode
)

( 
  steps/train_nnet_cpu_block.sh --cmd "queue.pl -pe smp 15" \
   --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4f_nnet

  steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
    --transform-dir exp/tri3b/decode \
    exp/tri3b/graph data/test exp/tri4f_nnet/decode
)


( 
  steps/train_nnet_cpu_block.sh --cmd "queue.pl -pe smp 15" \
   --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4g_nnet

  steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
    --transform-dir exp/tri3b/decode \
    exp/tri3b/graph data/test exp/tri4g_nnet/decode
)

( # as 4e (1m parameters), but realigning on the 2nd iter.
  steps/train_nnet_cpu.sh --cmd "$decode_cmd" --parallel-opts "-pe smp 15" --realign-iters "2" \
  --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4h_nnet

 steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
   --transform-dir exp/tri3b/decode \
   exp/tri3b/graph data/test exp/tri4h_nnet/decode
)

( # as 4h but realigning on the 4th iter, not the 2nd.
  # Realigning does not really seem to help.
  steps/train_nnet_cpu.sh --cmd "$decode_cmd" --parallel-opts "-pe smp 15" --realign-iters "4" \
  --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4i_nnet

 steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
   --transform-dir exp/tri3b/decode \
   exp/tri3b/graph data/test exp/tri4i_nnet/decode
)


( # 4j is as 4i, but running again after I changed the way the
  # validation set is selected (more utts, subset of frames.)
  steps/train_nnet_cpu.sh --cmd "$decode_cmd" --parallel-opts "-pe smp 15" --realign-iters "4" \
  --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4j_nnet

 steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
   --transform-dir exp/tri3b/decode \
   exp/tri3b/graph data/test exp/tri4j_nnet/decode
)

( # 4k is as 4j, but with --learning-rate-ratio=1.0 so it never decreases 
  # learning rate.
  steps/train_nnet_cpu.sh --cmd "$decode_cmd" --parallel-opts "-pe smp 15" --realign-iters "4" \
  --learning-rate-ratio 1.0 --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4k_nnet

 steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
   --transform-dir exp/tri3b/decode \
   exp/tri3b/graph data/test exp/tri4k_nnet/decode
)

( # 4l is as 4j, but with --measure-gradient-at 0.6 for faster learning.
  steps/train_nnet_cpu.sh --cmd "$decode_cmd" --parallel-opts "-pe smp 15" --realign-iters "4" \
  --measure-gradient-at 0.6 --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4l_nnet

 steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
   --transform-dir exp/tri3b/decode \
   exp/tri3b/graph data/test exp/tri4l_nnet/decode
)


( # 4m is as 4l, but --measure-gradient-at now 0.8 not 0.6.
  steps/train_nnet_cpu.sh --cmd "$decode_cmd" --parallel-opts "-pe smp 15" --realign-iters "4" \
  --measure-gradient-at 0.8 --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4m_nnet

 steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
   --transform-dir exp/tri3b/decode \
   exp/tri3b/graph data/test exp/tri4m_nnet/decode
)

( # 4n is as 4l, but --measure-gradient-at now 0.55 not 0.6.
  steps/train_nnet_cpu.sh --cmd "$decode_cmd" --parallel-opts "-pe smp 15" --realign-iters "4" \
  --measure-gradient-at 0.55 --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4n_nnet

 steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
   --transform-dir exp/tri3b/decode \
   exp/tri3b/graph data/test exp/tri4n_nnet/decode
)

( # 4o is as 4m, but changing initial l2 penalty to 0.
  steps/train_nnet_cpu.sh --nnet-config-opts "--l2-penalty 0.0" \
    --cmd "$decode_cmd" --parallel-opts "-pe smp 15" --realign-iters "4" \
    --measure-gradient-at 0.8 --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4o_nnet

 steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
   --transform-dir exp/tri3b/decode \
   exp/tri3b/graph data/test exp/tri4o_nnet/decode
)
( # 4o2 is as 4o, but actually doing what we said we were doing (fixed bug)
  steps/train_nnet_cpu.sh --nnet-config-opts "--l2-penalty 0.0" \
    --cmd "$decode_cmd" --parallel-opts "-pe smp 15" --realign-iters "4" \
    --measure-gradient-at 0.8 --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4o2_nnet

 steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
   --transform-dir exp/tri3b/decode \
   exp/tri3b/graph data/test exp/tri4o2_nnet/decode
)

( # 4p is as 4o (no l2 penalty, which is actually how all the previous expts were
  # but now making it explicit as we may have fixed the bug which caused that)
  # *and* smaller minibatch size; changing minibatches-per-phase to compensate.
  steps/train_nnet_cpu.sh --nnet-config-opts "--l2-penalty 0.0" \
    --minibatch-size 250 --minibatches-per-phase-it1 200 --minibatches-per-phase 800 \
    --cmd "$decode_cmd" --parallel-opts "-pe smp 15" --realign-iters "4" \
    --measure-gradient-at 0.8 --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4p_nnet

 steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
   --transform-dir exp/tri3b/decode \
   exp/tri3b/graph data/test exp/tri4p_nnet/decode
)


( # 4q is as 4p, but the binaries and script are now changed so bias-stddev is 1.0.
 # [not replicable now, use --bias-stddev 1.0]
  steps/train_nnet_cpu.sh --nnet-config-opts "--l2-penalty 0.0" \
    --minibatch-size 250 --minibatches-per-phase-it1 200 --minibatches-per-phase 800 \
    --cmd "$decode_cmd" --parallel-opts "-pe smp 15" --realign-iters "4" \
    --measure-gradient-at 0.8 --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4q_nnet

 steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
   --transform-dir exp/tri3b/decode \
   exp/tri3b/graph data/test exp/tri4q_nnet/decode
)

( # 4r is as 4q, but the binaries and script are now changed so bias-stddev is 2.0 by default.
  steps/train_nnet_cpu.sh --nnet-config-opts "--l2-penalty 0.0" \
    --minibatch-size 250 --minibatches-per-phase-it1 200 --minibatches-per-phase 800 \
    --cmd "$decode_cmd" --parallel-opts "-pe smp 15" --realign-iters "4" \
    --measure-gradient-at 0.8 --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4r_nnet

 steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
   --transform-dir exp/tri3b/decode \
   exp/tri3b/graph data/test exp/tri4r_nnet/decode
)

( # 4s is as 4r but bias-stddev is 4.0 not 2.0.
  steps/train_nnet_cpu.sh --nnet-config-opts "--l2-penalty 0.0 --bias-stddev 4.0" \
    --minibatch-size 250 --minibatches-per-phase-it1 200 --minibatches-per-phase 800 \
    --cmd "$decode_cmd" --parallel-opts "-pe smp 15" --realign-iters "4" \
    --measure-gradient-at 0.8 --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4s_nnet

 steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
   --transform-dir exp/tri3b/decode \
   exp/tri3b/graph data/test exp/tri4s_nnet/decode
)

( # 4t is as 4r but rerunning-- code should not have changed but we
 # want to check. [yes, it was the same.]
  steps/train_nnet_cpu.sh --nnet-config-opts "--l2-penalty 0.0" \
    --minibatch-size 250 --minibatches-per-phase-it1 200 --minibatches-per-phase 800 \
    --cmd "$decode_cmd" --parallel-opts "-pe smp 15" --realign-iters "4" \
    --measure-gradient-at 0.8 --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4t_nnet

 steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
   --transform-dir exp/tri3b/decode \
   exp/tri3b/graph data/test exp/tri4t_nnet/decode
)

( # 4v is as 4u but with code changed to do it more "properly".. previously was not working.
  steps/train_nnet_cpu.sh --nnet-config-opts "--l2-penalty 0.0 --precondition true" \
    --minibatch-size 250 --minibatches-per-phase-it1 200 --minibatches-per-phase 800 \
    --cmd "$decode_cmd" --parallel-opts "-pe smp 15" --realign-iters "4" \
    --measure-gradient-at 0.8 --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4v_nnet

 steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
   --transform-dir exp/tri3b/decode \
   exp/tri3b/graph data/test exp/tri4v_nnet/decode
)


( # 4w is as 4v but rerunning after I changed the code to remove l2 regularization;
  # just checking it still runs.
  steps/train_nnet_cpu.sh --nnet-config-opts "--precondition true" \
    --minibatch-size 250 --minibatches-per-phase-it1 200 --minibatches-per-phase 800 \
    --cmd "$decode_cmd" --parallel-opts "-pe smp 15" --realign-iters "4" \
    --measure-gradient-at 0.8 --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4w_nnet

 steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
   --transform-dir exp/tri3b/decode \
   exp/tri3b/graph data/test exp/tri4w_nnet/decode
)

( # running more iters on 4w.
  steps/train_nnet_cpu.sh --nnet-config-opts "--precondition true" \
    --minibatch-size 250 --minibatches-per-phase-it1 200 --minibatches-per-phase 800 \
    --cmd "$decode_cmd" --parallel-opts "-pe smp 15" --realign-iters "4" \
    --measure-gradient-at 0.8 --num-parameters 1000000  data/train data/lang exp/tri3b_ali exp/tri4w_nnet

 steps/decode_nnet_cpu.sh --cmd "$decode_cmd" --nj 20 \
   --transform-dir exp/tri3b/decode \
   exp/tri3b/graph data/test exp/tri4w_nnet/decode
)
