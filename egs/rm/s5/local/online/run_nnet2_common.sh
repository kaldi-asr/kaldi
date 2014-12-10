#!/bin/bash

. cmd.sh


stage=1

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
  dir=exp/nnet2_online/nnet
fi


if [ $stage -le 1 ]; then
  mkdir -p exp/nnet2_online
  steps/online/nnet2/train_diag_ubm.sh --cmd "$train_cmd" --nj 10 --num-frames 200000 \
    data/train 256 exp/tri3b exp/nnet2_online/diag_ubm
fi

if [ $stage -le 2 ]; then
  # use a smaller iVector dim (50) than the default (100) because RM has a very
  # small amount of data.
  steps/online/nnet2/train_ivector_extractor.sh --cmd "$train_cmd" --nj 4 \
    --ivector-dim 50 \
   data/train exp/nnet2_online/diag_ubm exp/nnet2_online/extractor || exit 1;
fi

if [ $stage -le 3 ]; then
  # having a larger number of speakers is helpful for generalization, and to
  # handle per-utterance decoding well (iVector starts at zero).
  steps/online/nnet2/copy_data_dir.sh --utts-per-spk-max 2 data/train data/train_max2

  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 4 \
    data/train_max2 exp/nnet2_online/extractor exp/nnet2_online/ivectors || exit 1;
fi
