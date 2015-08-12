#!/bin/bash


# This runs on the full training set, with maxout
# units, on top of fMLLR features, on GPU.

temp_dir=
dir=exp/nnet2_maxout
stage=-4

. ./cmd.sh
. ./path.sh

. utils/parse_options.sh

parallel_opts="--gpu 1"  # This is suitable for the CLSP network, you'll
                          # likely have to change it.

( 
  if [ ! -f $dir/final.mdl ]; then
    steps/nnet2/train_maxout_accel2.sh --parallel-opts "$parallel_opts" \
      --cmd "$decode_cmd" --stage $stage \
      --num-threads 1 --minibatch-size 512 \
      --mix-up 20000 --samples-per-iter 300000 \
      --num-epochs 15 \
      --initial-effective-lrate 0.005 --final-effective-lrate 0.0005 \
      --num-jobs-initial 3 --num-jobs-final 8 --num-hidden-layers 4 --splice-width 5 \
      --maxout-input-dim 2400 --maxout-output-dim 800 \
      data/train data/lang exp/tri5a_ali $dir || exit 1;
  fi

  steps/nnet2/decode.sh --cmd "$decode_cmd" --nj 10 \
    --config conf/decode.config \
    --transform-dir exp/tri5a/decode \
    exp/tri5a/graph data/dev \
    $dir/decode || exit 1;
)
