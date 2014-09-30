#!/bin/bash


# this is a baseline for ./run_nnet2.sh, without
# the iVectors, to see whether they make a difference.

. cmd.sh


stage=1
train_stage=-10
use_gpu=true
dir=exp/nnet2_online/nnet_a_baseline

. cmd.sh
. ./path.sh
. ./utils/parse_options.sh



if $use_gpu; then
  if ! cuda-compiled; then
    cat <<EOF && exit 1 
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA 
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
  fi
  parallel_opts="-l gpu=1" 
  num_threads=1
  minibatch_size=512
else
  # Use 4 nnet jobs just like run_4d_gpu.sh so the results should be
  # almost the same, but this may be a little bit slow.
  num_threads=16
  minibatch_size=128
  parallel_opts="-pe smp $num_threads" 
fi


if [ $stage -le 1 ]; then
  steps/nnet2/train_pnorm_simple.sh --stage $train_stage \
    --splice-width 7 \
    --feat-type raw \
    --cmvn-opts "--norm-means=false --norm-vars=false" \
    --num-threads "$num_threads" \
    --minibatch-size "$minibatch_size" \
    --parallel-opts "$parallel_opts" \
    --num-jobs-nnet 4 \
    --num-epochs 25 \
    --add-layers-period 1 \
    --num-hidden-layers 2 \
    --mix-up 4000 \
    --initial-learning-rate 0.02 --final-learning-rate 0.004 \
    --cmd "$decode_cmd" \
    --pnorm-input-dim 1000 \
    --pnorm-output-dim 200 \
    data/train data/lang exp/tri3b_ali $dir  || exit 1;
fi

if [ $stage -le 2 ]; then
  steps/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 20 \
    exp/tri3b/graph data/test $dir/decode  &

  steps/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 20 \
    exp/tri3b/graph_ug data/test $dir/decode_ug || exit 1;

  wait
fi

if [ $stage -le 3 ]; then
  # If this setup used PLP features, we'd have to give the option --feature-type plp
  # to the script below.
  steps/online/nnet2/prepare_online_decoding.sh data/lang "$dir" ${dir}_online || exit 1;
fi


if [ $stage -le 4 ]; then
  # Doing the real online decoding.  The --per-utt true option actually
  # makes no difference to the output as there is no adaptation at all.
  steps/online/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 20 \
    --per-utt true exp/tri3b/graph data/test ${dir}_online/decode &
  steps/online/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 20 \
    --per-utt true exp/tri3b/graph_ug data/test ${dir}_online/decode_ug || exit 1;
  wait
fi

# for results, see the end of ./run_nnet2.sh
