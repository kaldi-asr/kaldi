#!/usr/bin/env bash

. ./cmd.sh
set -e
stage=1
train_stage=-10
use_gpu=true
splice_indexes="layer0/-2:-1:0:1:2 layer1/-1:2 layer2/-3:3 layer3/-7:2 layer4/-3:3"
common_egs_dir=
dir=exp/nnet2_online/nnet_ms

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
  parallel_opts="--gpu 1"
  num_threads=1
  minibatch_size=512
  # the _a is in case I want to change the parameters.
else
  # Use 4 nnet jobs just like run_4d_gpu.sh so the results should be
  # almost the same, but this may be a little bit slow.
  num_threads=16
  minibatch_size=128
  parallel_opts="--num-threads $num_threads"
fi

# Run the common stages of training, including training the iVector extractor
local/online/run_nnet2_common.sh --stage $stage || exit 1;

if [ $stage -le 6 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] &&\
    [ ! -d $dir/egs/storage ] && [ -z $common_egs_dir ]; then
    utils/create_split_dir.pl \
     /export/b0{5,6,7,8}/$USER/kaldi-data/egs/hkust-$(date +'%m_%d_%H_%M')/s5c/$dir/egs/storage $dir/egs/storage
  fi

  steps/nnet2/train_multisplice_accel2.sh --stage $train_stage \
    --num-epochs 4 --num-jobs-initial 3 --num-jobs-final 8 \
    --num-hidden-layers 5 --splice-indexes "$splice_indexes" \
    --feat-type raw \
    --online-ivector-dir exp/nnet2_online/ivectors_train \
    --cmvn-opts "--norm-means=false --norm-vars=false" \
    --num-threads "$num_threads" \
    --minibatch-size "$minibatch_size" \
    --parallel-opts "$parallel_opts" \
    --io-opts "--max-jobs-run 12" \
    --add-layers-period 1 \
    --mix-up 20000 \
    --initial-effective-lrate 0.0015 --final-effective-lrate 0.00015 \
    --cmd "$decode_cmd" \
    --egs-dir "$common_egs_dir" \
    --pnorm-input-dim 4000 \
    --pnorm-output-dim 400 \
    data/train_hires data/lang exp/tri5a_ali $dir  || exit 1;
fi

if [ $stage -le 7 ]; then
  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 10 \
    data/dev_hires_nopitch exp/nnet2_online/extractor \
    exp/nnet2_online/ivectors_dev || exit 1;
fi

if [ $stage -le 8 ]; then
  # this does offline decoding that should give the same results as the real
  # online decoding (the one with --per-utt true)
  graph_dir=exp/tri5a/graph
  # use already-built graphs.
  steps/nnet2/decode.sh --nj 10 --cmd "$decode_cmd" \
    --config conf/decode.config \
    --online-ivector-dir exp/nnet2_online/ivectors_dev \
    $graph_dir data/dev_hires $dir/decode || exit 1;
fi

if [ $stage -le 9 ]; then
  steps/online/nnet2/prepare_online_decoding.sh --mfcc-config conf/mfcc_hires.conf \
    --add-pitch true \
    data/lang exp/nnet2_online/extractor "$dir" ${dir}_online || exit 1;
fi

if [ $stage -le 10 ]; then
  # do the actual online decoding with iVectors, carrying info forward from
  # previous utterances of the same speaker.
  graph_dir=exp/tri5a/graph
  steps/online/nnet2/decode.sh --config conf/decode.config \
    --cmd "$decode_cmd" --nj 10 \
    "$graph_dir" data/dev_hires \
    ${dir}_online/decode || exit 1;
fi

if [ $stage -le 11 ]; then
  # this version of the decoding treats each utterance separately
  # without carrying forward speaker information.
  graph_dir=exp/tri5a/graph
  steps/online/nnet2/decode.sh --config conf/decode.config \
    --cmd "$decode_cmd" --nj 10 --per-utt true \
    "$graph_dir" data/dev_hires \
    ${dir}_online/decode_per_utt || exit 1;
fi

exit 0;
