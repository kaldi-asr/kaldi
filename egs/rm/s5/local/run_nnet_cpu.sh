#!/bin/bash

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

# should be same as 4c.
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

