#!/bin/bash

. cmd.sh


stage=1
train_stage=-10
use_gpu=true
dir=exp/nnet2_online/nnet_ms_a


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
else
  # Use 4 nnet jobs just like run_4d_gpu.sh so the results should be
  # almost the same, but this may be a little bit slow.
  num_threads=16
  minibatch_size=128
  parallel_opts="-pe smp $num_threads" 
fi



# stages 1 through 3 run in run_nnet2_common.sh.

local/online/run_nnet2_common.sh --stage  $stage || exit 1;


if [ $stage -le 4 ]; then
  steps/nnet2/train_multisplice_accel2.sh --stage $train_stage \
    --splice-indexes "layer0/-2:-1:0:1:2 layer1/-3:1 layer2/-5:3" \
    --num-hidden-layers 3 \
    --feat-type raw \
    --online-ivector-dir exp/nnet2_online/ivectors \
    --cmvn-opts "--norm-means=false --norm-vars=false" \
    --num-threads "$num_threads" \
    --minibatch-size "$minibatch_size" \
    --parallel-opts "$parallel_opts" \
    --num-jobs-initial 2 --num-jobs-final 4 \
    --num-epochs 25 \
    --add-layers-period 1 \
    --mix-up 4000 \
    --initial-effective-lrate 0.005 --final-effective-lrate 0.0005 \
    --cmd "$decode_cmd" \
    --pnorm-input-dim 800 \
    --pnorm-output-dim 160 \
    data/train data/lang exp/tri3b_ali $dir  || exit 1;
fi

if [ $stage -le 5 ]; then
  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 4 \
    data/test exp/nnet2_online/extractor exp/nnet2_online/ivectors_test || exit 1;
fi


if [ $stage -le 6 ]; then
  # Note: comparing the results of this with run_online_decoding_nnet2_baseline.sh,
  # it's a bit worse, meaning the iVectors seem to hurt at this amount of data.
  # However, experiments by Haihua Xu (not checked in yet) on WSJ, show it helping
  # nicely.  This setup seems to have too little data for it to work, but it suffices
  # to demonstrate the scripts.   We will likely modify it to add noise to the
  # iVectors in training, which will tend to mitigate the over-training.
  steps/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 20 \
    --online-ivector-dir exp/nnet2_online/ivectors_test \
    exp/tri3b/graph data/test $dir/decode  &

  steps/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 20 \
    --online-ivector-dir exp/nnet2_online/ivectors_test \
    exp/tri3b/graph_ug data/test $dir/decode_ug || exit 1;

  wait
fi

if [ $stage -le 7 ]; then
  # If this setup used PLP features, we'd have to give the option --feature-type plp
  # to the script below.
  steps/online/nnet2/prepare_online_decoding.sh data/lang exp/nnet2_online/extractor \
    "$dir" ${dir}_online || exit 1;
fi

if [ $stage -le 8 ]; then
  # do the actual online decoding with iVectors.
  steps/online/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 20 \
    exp/tri3b/graph data/test ${dir}_online/decode &
  steps/online/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 20 \
    exp/tri3b/graph_ug data/test ${dir}_online/decode_ug || exit 1;
  wait
fi

if [ $stage -le 9 ]; then
  # this version of the decoding treats each utterance separately
  # without carrying forward speaker information.
  steps/online/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 20 \
    --per-utt true \
    exp/tri3b/graph data/test ${dir}_online/decode_per_utt &
  steps/online/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 20 \
    --per-utt true \
    exp/tri3b/graph_ug data/test ${dir}_online/decode_ug_per_utt || exit 1;
  wait
fi

exit 0;



# see ../../RESULTS for results.  It's about the same as the non-multisplice
# recipe, but I'm not doing much tuning on RM... it has too little data 
# for any of these DNN things to really work well


