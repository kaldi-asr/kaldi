#!/usr/bin/env bash


# This runs on the full training set (with duplicates removed), with p-norm units, on top of fMLLR features, on GPU.
# This version uses train_pnorm_fast.sh, rather than the old version train_pnorm.sh

temp_dir=
dir=nnet5f_gpu
. ./cmd.sh
. ./path.sh
. utils/parse_options.sh
parallel_opts="--gpu 1"  # This is suitable for the CLSP network, you'll likely have to change it.

( 
   if [ ! -f exp/$dir/final.mdl ]; then
     if [ ! -z "$temp_dir" ] && [ ! -e exp/$dir/egs ]; then
       mkdir -p exp/$dir
       mkdir -p $temp_dir/$dir/egs
       ln -s $temp_dir/$dir/egs exp/$dir/
     fi

     steps/nnet2/train_pnorm_fast.sh --parallel-opts "$parallel_opts" \
       --cmd "$decode_cmd" \
       --stage -10 \
       --num-threads 1 --minibatch-size 512 --mix-up 20000 --samples-per-iter 300000 \
       --num-epochs 10 --num-epochs-extra 5 --initial-learning-rate 0.05 --final-learning-rate 0.002 \
       --num-jobs-nnet 10 --num-hidden-layers 5 --pnorm-input-dim 5000  --pnorm-output-dim 500 data/train_nodup \
       data/lang exp/tri4b exp/$dir || exit 1;
   fi

  for lm_suffix in tg fsh_tgpr; do
    steps/nnet2/decode.sh --cmd "$decode_cmd" --nj 30 \
      --config conf/decode.config --transform-dir exp/tri4b/decode_eval2000_sw1_${lm_suffix} \
      exp/tri4b/graph_sw1_${lm_suffix} data/eval2000 exp/$dir/decode_eval2000_sw1_${lm_suffix} &
  done
)
