#!/usr/bin/env bash

. ./cmd.sh


stage=1
train_stage=-10
use_gpu=true
set -e
. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh


# assume use_gpu=true since it would be way too slow otherwise.

if ! cuda-compiled; then
  cat <<EOF && exit 1 
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA 
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi
parallel_opts="--gpu 1" 
num_threads=1
minibatch_size=512
dir=exp/nnet2_online/nnet_a
mkdir -p exp/nnet2_online


# Stages 1 through 5 are done in run_nnet2_common.sh,
# so it can be shared with other similar scripts.
local/online/run_nnet2_common.sh --stage $stage

if [ $stage -le 6 ]; then
  if [ $USER == dpovey ]; then # this shows how you can split across multiple file-systems.
    utils/create_split_dir.pl /export/b0{3,4,5,6}/dpovey/kaldi-online/egs/fisher_english/s5/$dir/egs $dir/egs/storage
  fi

  # Because we have a lot of data here and we don't want the training to take
  # too long, we reduce the number of epochs from the defaults (15 + 5) to (3 +
  # 1).  The option "--io-opts '--max-jobs-run 12'" is to have more than the default number
  # (5) of jobs dumping the egs to disk; this is OK since we're splitting our
  # data across four filesystems for speed.

  steps/nnet2/train_pnorm_simple2.sh --stage $train_stage \
    --num-epochs 5 \
    --splice-width 7 \
    --feat-type raw \
    --online-ivector-dir exp/nnet2_online/ivectors_train \
    --cmvn-opts "--norm-means=false --norm-vars=false" \
    --num-threads "$num_threads" \
    --minibatch-size "$minibatch_size" \
    --parallel-opts "$parallel_opts" \
    --io-opts "--max-jobs-run 12" \
    --num-jobs-nnet 6 \
    --num-hidden-layers 4 \
    --mix-up 12000 \
    --initial-learning-rate 0.01 --final-learning-rate 0.001 \
    --cmd "$decode_cmd" \
    --pnorm-input-dim 3500 \
    --pnorm-output-dim 350 \
    data/train_hires data/lang exp/tri5a $dir  || exit 1;
fi

if [ $stage -le 7 ]; then
  steps/online/nnet2/prepare_online_decoding.sh --mfcc-config conf/mfcc_hires.conf \
    data/lang exp/nnet2_online/extractor "$dir" ${dir}_online || exit 1;
fi

if [ $stage -le 8 ]; then
  # do the actual online decoding with iVectors, carrying info forward from 
  # previous utterances of the same speaker.
   steps/online/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 30 \
      exp/tri5a/graph data/dev ${dir}_online/decode_dev || exit 1;
fi

if [ $stage -le 9 ]; then
  # this version of the decoding treats each utterance separately
  # without carrying forward speaker information.
   steps/online/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 30 \
     --per-utt true \
      exp/tri5a/graph data/dev ${dir}_online/decode_dev_utt || exit 1;
fi

if [ $stage -le 10 ]; then
  # this version of the decoding treats each utterance separately
  # without carrying forward speaker information, but looks to the end
  # of the utterance while computing the iVector.
   steps/online/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 30 \
     --per-utt true --online false \
      exp/tri5a/graph data/dev ${dir}_online/decode_dev_utt_offline || exit 1;
fi

exit 0;
