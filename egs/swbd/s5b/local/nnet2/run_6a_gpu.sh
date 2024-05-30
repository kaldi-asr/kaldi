#!/usr/bin/env bash

# This runs on the 100 hour subset; it's another neural-net training 
# after the nnet5a setup, but after realignment.   We're just seeing
# whether realigning and then re-training the system is helpful.
#
# e.g. of usage:
# local/nnet2/run_6a_gpu.sh --temp-dir /export/gpu-03/dpovey/kaldi-dan2/egs/swbd/s5b

temp_dir=
train_stage=-10

. ./cmd.sh
. ./path.sh
! cuda-compiled && cat <<EOF && exit 1 
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA 
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF


. utils/parse_options.sh

parallel_opts="--gpu 1"  # This is suitable for the CLSP network, you'll likely have to change it.

alidir=exp/nnet5a_ali_100k_nodup
if [ ! -f $alidir/.done ]; then
  nj=`cat exp/tri4a/num_jobs`
  steps/nnet2/align.sh --cmd "$decode_cmd" --nj $nj --transform-dir exp/tri4a \
      data/train_100k_nodup data/lang exp/nnet5a_gpu $alidir || exit 1;
  touch $alidir/.done
fi

if [ ! -f exp/nnet6a_gpu/final.mdl ]; then
  if [ ! -z "$temp_dir" ] && [ ! -e exp/nnet6a_gpu/egs ]; then
    mkdir -p exp/nnet6a_gpu
    mkdir -p $temp_dir/nnet6a_gpu/egs
    ln -s $temp_dir/nnet6a_gpu/egs exp/nnet6a_gpu/
  fi

  # TODO: add transform-dir option to train_tanh.sh
  steps/nnet2/train_tanh.sh --stage $train_stage \
    --num-jobs-nnet 8 --num-threads 1 --max-change 40.0 \
    --minibatch-size 512 --parallel-opts "$parallel_opts" \
    --mix-up 8000 \
    --initial-learning-rate 0.01 --final-learning-rate 0.001 \
    --num-hidden-layers 4 \
    --hidden-layer-dim 1024 \
    --cmd "$decode_cmd" \
    --egs-opts "--transform-dir exp/tri4a" \
    data/train_100k_nodup data/lang $alidir exp/nnet6a_gpu || exit 1;
fi

for lm_suffix in tg fsh_tgpr; do
  steps/nnet2/decode.sh --cmd "$decode_cmd" --nj 30 \
    --config conf/decode.config --transform-dir exp/tri4a/decode_eval2000_sw1_${lm_suffix} \
    exp/tri4a/graph_sw1_${lm_suffix} data/eval2000 exp/nnet6a_gpu/decode_eval2000_sw1_${lm_suffix} &
done


