#!/usr/bin/env bash

# this is as 5d (100 hour subset, pnorm),
# but on top of the raw-fMLLR input
# Results were worse than 5d when I tested it but both the script and baseline
# have changed since then.  I'm giving parameters that I think are reasonable,
# but they need to be tested.


# e.g. of usage:
# local/nnet2/run_5d_gpu.sh --temp-dir /export/m1-01/dpovey/kaldi-dan2/egs/swbd/s5b

dir=nnet5e_gpu
temp_dir=
train_stage=-10

. ./cmd.sh


. utils/parse_options.sh
parallel_opts="--gpu 1"  # This is suitable for the CLSP network, you'll likely have to change it.

( 
  if [ ! -f exp/$dir/final.mdl ]; then
    if [ ! -z "$temp_dir" ] && [ ! -e exp/$dir/egs ]; then
      mkdir -p exp/$dir
      mkdir -p $temp_dir/$dir/egs
      ln -s $temp_dir/$dir/egs exp/$dir/
    fi

    steps/nnet2/train_pnorm.sh --stage $train_stage --splice-width 6 \
      --num-jobs-nnet 8 --num-threads 1 --max-change 40.0 \
      --num-epochs 10 \
      --minibatch-size 512 --parallel-opts "$parallel_opts" \
      --mix-up 8000 \
      --initial-learning-rate 0.08 --final-learning-rate 0.008 \
      --num-hidden-layers 4 \
      --pnorm-input-dim 3000 \
      --pnorm-output-dim 300 \
      --cmd "$decode_cmd" \
      data/train_100k_nodup data/lang exp/tri4d exp/$dir || exit 1;
  fi

  for lm_suffix in tg fsh_tgpr; do
    steps/nnet2/decode.sh --cmd "$decode_cmd" --nj 30 \
      --config conf/decode.config --transform-dir exp/tri4d/decode_eval2000_sw1_${lm_suffix} \
      exp/tri4d/graph_sw1_${lm_suffix} data/eval2000 exp/$dir/decode_eval2000_sw1_${lm_suffix} &
  done
)

