#!/bin/bash

# 2015 Xingyu Na
# This script runs on the full training set, using ConvNet setup on top of
# fbank features, on GPU. The ConvNet has four hidden layers, two convolutional
# layers and two affine transform layers with ReLU nonlinearity.
# Convolutional layer [1]:
#   convolution1d, input feature dim is 36, filter dim is 7, output dim is
#   30, 128 filters are used
#   maxpooling, 3-to-1 maxpooling, input dim is 30, output dim is 10
# Convolutional layer [2]:
#   convolution1d, input feature dim is 10, filter dim is 4, output dim is
#   7, 256 filters are used
# Affine transform layers [3-4]:
#   affine transform with ReLU nonlinearity.

temp_dir=
dir=exp/nnet2_convnet
stage=-5
train_original=data/train
train=data-fb/train

. ./cmd.sh
. ./path.sh

. utils/parse_options.sh

parallel_opts="--gpu 1"  # This is suitable for the CLSP network, you'll
                         # likely have to change it.

# Make the FBANK features
if [ $stage -le -5 ]; then
  # Dev set
  utils/copy_data_dir.sh data/dev data-fb/dev || exit 1; rm $train/{cmvn,feats}.scp
  steps/make_fbank.sh --nj 10 --cmd "$train_cmd" \
     data-fb/dev data-fb/dev/log data-fb/dev/data || exit 1;
  steps/compute_cmvn_stats.sh data-fb/dev data-fb/dev/log data-fb/dev/data || exit 1;
  # Training set
  utils/copy_data_dir.sh $train_original $train || exit 1; rm $train/{cmvn,feats}.scp
  steps/make_fbank.sh --nj 10 --cmd "$train_cmd" \
     $train $train/log $train/data || exit 1;
  steps/compute_cmvn_stats.sh $train $train/log $train/data || exit 1;
fi

( 
  if [ ! -f $dir/final.mdl ]; then
    steps/nnet2/train_convnet_accel2.sh --parallel-opts "$parallel_opts" \
      --cmd "$decode_cmd" --stage $stage \
      --num-threads 1 --minibatch-size 512 \
      --mix-up 20000 --samples-per-iter 300000 \
      --num-epochs 15 --delta-order 2 \
      --initial-effective-lrate 0.0001 --final-effective-lrate 0.00001 \
      --num-jobs-initial 3 --num-jobs-final 8 --splice-width 5 \
      --hidden-dim 2000 --num-filters1 128 --patch-dim1 7 --pool-size 3 \
      --num-filters2 256 --patch-dim2 4 \
      $train data/lang exp/tri5a_ali $dir || exit 1;
  fi

  steps/nnet2/decode.sh --cmd "$decode_cmd" --nj 10 \
    --config conf/decode.config \
    exp/tri5a/graph data-fb/dev \
    $dir/decode || exit 1;
)
