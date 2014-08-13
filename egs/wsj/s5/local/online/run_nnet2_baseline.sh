#!/bin/bash

. cmd.sh


stage=1
train_stage=-10
use_gpu=true
. cmd.sh
. ./path.sh
. ./utils/parse_options.sh

if $use_gpu; then
  if ! cuda-compiled; then
    cat <<EOF && exit 1 
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA 
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.  Otherwise, call this script with --use-gpu false
EOF
  fi
  parallel_opts="-l gpu=1" 
  num_threads=1
  minibatch_size=512
  # the _a is in case I want to change the parameters.
  dir=exp/nnet2_online/nnet_a_gpu_baseline
else
  # Use 4 nnet jobs just like run_4d_gpu.sh so the results should be
  # almost the same, but this may be a little bit slow.
  num_threads=16
  minibatch_size=128
  parallel_opts="-pe smp $num_threads" 
  dir=exp/nnet2_online/nnet_a_baseline
fi



if [ $stage -le 1 ]; then
  # train without iVectors.
  steps/nnet2/train_pnorm_fast.sh --stage $train_stage \
    --num-epochs 8 --num-epochs-extra 4 \
    --splice-width 7 --feat-type raw \
    --cmvn-opts "--norm-means=false --norm-vars=false" \
    --num-threads "$num_threads" \
    --minibatch-size "$minibatch_size" \
    --parallel-opts "$parallel_opts" \
    --num-jobs-nnet 6 \
    --num-hidden-layers 4 \
    --mix-up 4000 \
    --initial-learning-rate 0.02 --final-learning-rate 0.004 \
    --cmd "$decode_cmd" \
    --pnorm-input-dim 2400 \
    --pnorm-output-dim 300 \
    data/train_si284 data/lang exp/tri4b_ali_si284 $dir  || exit 1;
fi


if [ $stage -le 2 ]; then
  for lm_suffix in tgpr bd_tgpr; do
    graph_dir=exp/tri4b/graph_${lm_suffix}
    # use already-built graphs.
    for year in eval92 dev93; do
      steps/nnet2/decode.sh --nj 8 --cmd "$decode_cmd" \
         $graph_dir data/test_$year $dir/decode_${lm_suffix}_${year} || exit 1;
    done
  done
fi

if [ $stage -le 3 ]; then
  # If this setup used PLP features, we'd have to give the option --feature-type plp
  # to the script below.
  steps/online/nnet2/prepare_online_decoding.sh data/lang "$dir" ${dir}_online || exit 1;
fi


if [ $stage -le 4 ]; then
  # Decode.  The --per-utt true option makes no difference to the results here.
  for lm_suffix in tgpr bd_tgpr; do
    graph_dir=exp/tri4b/graph_${lm_suffix}
    for year in eval92 dev93; do
      steps/online/nnet2/decode.sh --cmd "$decode_cmd" --nj 8 \
        --per-utt true \
        "$graph_dir" data/test_${year} ${dir}_online/decode_${lm_suffix}_${year}_utt || exit 1;
    done
  done
fi

