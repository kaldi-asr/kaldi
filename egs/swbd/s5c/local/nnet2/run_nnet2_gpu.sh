#!/bin/bash


# This runs on the full training set (with duplicates removed), with p-norm
# units, on top of fMLLR features, on GPU.

temp_dir=
dir=nnet2_5_gpu
has_fisher=true

. ./cmd.sh
. ./path.sh

. utils/parse_options.sh

parallel_opts="-l gpu=1"  # This is suitable for the CLSP network, you'll
                          # likely have to change it.

( 
  if [ ! -f exp/$dir/final.mdl ]; then
    if [ ! -z "$temp_dir" ] && [ ! -e exp/$dir/egs ]; then
      mkdir -p exp/$dir
      mkdir -p $temp_dir/$dir/egs
      ln -s $temp_dir/$dir/egs exp/$dir/
    fi

    steps/nnet2/train_pnorm_fast.sh --parallel-opts "$parallel_opts" \
      --cmd "$decode_cmd" --stage -10 \
      --num-threads 1 --minibatch-size 512 \
      --mix-up 20000 --samples-per-iter 300000 \
      --num-epochs 10 --num-epochs-extra 5 \
      --initial-learning-rate 0.05 --final-learning-rate 0.002 \
      --num-jobs-nnet 10 --num-hidden-layers 5 \
      --pnorm-input-dim 5000  --pnorm-output-dim 500 data/train_nodup \
      data/lang exp/tri4 exp/$dir || exit 1;
  fi

  steps/nnet2/decode.sh --cmd "$decode_cmd" --nj 30 \
    --config conf/decode.config \
    --transform-dir exp/tri4/decode_eval2000_sw1_tg \
    exp/tri4/graph_sw1_tg data/eval2000 \
    exp/$dir/decode_eval2000_sw1_tg || exit 1;

  if [ $has_fisher ]; then
    steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
      data/lang_sw1_{tg,fsh_fg} data/eval2000 \
      exp/$dir/decode_eval2000_sw1_{tg,fsh_fg} || exit 1;
  fi
)
