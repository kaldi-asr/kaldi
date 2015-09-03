#!/bin/bash

# This is GPU based pnorm neural net ensemble training on top of adapted 40-dimensional features.

parallel_opts="-l gpu=1" 

. cmd.sh

dir=exp/nnet4e_gpu
(  steps/nnet2/train_pnorm_ensemble.sh  --num-epochs 20 \
     --num-jobs-nnet 4 --num-threads 1 --parallel-opts "$parallel_opts" \
     --num-epochs-extra 10 --add-layers-period 1 \
     --num-hidden-layers 2 \
     --mix-up 4000 \
     --initial-learning-rate 0.02 --final-learning-rate 0.004 \
     --cmd "$decode_cmd" \
     --pnorm-input-dim 1000 \
     --pnorm-output-dim 200 \
     --combine-regularizer 1.0e-12 \
     --ensemble-size 4 --initial-beta 0.1 --final-beta 5 \
     data/train data/lang exp/tri3b_ali $dir 

   steps/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 20 \
     --transform-dir exp/tri3b/decode \
     exp/tri3b/graph data/test $dir/decode 

   steps/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 20 \
     --transform-dir exp/tri3b/decode_ug \
     exp/tri3b/graph_ug data/test $dir/decode_ug

)


