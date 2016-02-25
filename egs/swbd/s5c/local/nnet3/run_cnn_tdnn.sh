#!/bin/bash

# Copyright 2015  Johns Hopkins University (Author: Daniel Povey).
#           2015  Vijayaditya Peddinti
#           2016  Tom Ko
# Apache 2.0.

# This is the CNN+TDNN system built in nnet3
# This system differs from the standard TDNN system
# in replacing the LDA with CNN at the front of the network.
# and using fbank features as the CNN input
# As the dimension of the CNN output is usually large, we place
# a linear layer at the output of CNN for dimension reduction
# The ivectors are processed through a fully connected affine layer,
# then concatenated with the dimension-reduced CNN output and 
# passed to the deeper part of the network.
# Due to the data compression issue, it is better to convert MFCC
# to FBANK in the network instead of directly using FBANK features
# from the storage. This script uses MFCC features as its input
# and were converted to FBANK features with an inverse of DCT matrix
# at the first layer of the network.


# At this script level we don't support not running on GPU, as it would be painfully slow.
# If you want to run without GPU you'd have to call train_tdnn.sh with --gpu false,
# --num-threads 16 and --minibatch-size 128.

stage=9
train_stage=-10
has_fisher=true
speed_perturb=true

# CNN options
# Parameter indices used for each CNN layer
# Format: layer<CNN_index>/<parameter_indices>....layer<CNN_index>/<parameter_indices>
# The <parameter_indices> for each CNN layer must contain 11 positive integers.
# The first 5 integers correspond to the parameter of ConvolutionComponent:
# <filt_x_dim, filt_y_dim, filt_x_step, filt_y_step, num_filters>
# The next 6 integers correspond to the parameter of MaxpoolingComponent:
# <pool_x_size, pool_y_size, pool_z_size, pool_x_step, pool_y_step, pool_z_step>
cnn_indexes="3,8,1,1,256,1,3,1,1,3,1"
# Output dimension of the linear layer at the CNN output for dimension reduction
cnn_reduced_dim=256
# Choose whether to generate delta and delta-delta features
# by adding a fixed convolution layer
conv_add_delta=false

splice_indexes="-2,-1,0,1,2 -1,2 -3,3 -7,2 0"

. cmd.sh
. ./path.sh
. ./utils/parse_options.sh


if ! cuda-compiled; then
  cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi

suffix=
if [ "$speed_perturb" == "true" ]; then
  suffix=_sp
fi
dir=exp/nnet3/cnn_test
dir=$dir${affix:+_$affix}
dir=${dir}$suffix
train_set=train_nodup$suffix
ali_dir=exp/tri4_ali_nodup$suffix

local/nnet3/run_ivector_common.sh --stage $stage \
	--speed-perturb $speed_perturb || exit 1;

if [ $stage -le 9 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{3,4,5,6}/$USER/kaldi-data/egs/swbd-$(date +'%m_%d_%H_%M')/s5/$dir/egs/storage $dir/egs/storage
  fi

  steps/nnet3/train_cnn_tdnn.sh --stage $train_stage \
    --num-epochs 2 --num-jobs-initial 3 --num-jobs-final 16 \
    --cnn-indexes "$cnn_indexes" \
    --cnn-reduced-dim $cnn_reduced_dim \
    --conv-add-delta $conv_add_delta \
    --use-mfcc "true" \
    --splice-indexes "$splice_indexes" \
    --online-ivector-dir exp/nnet3/ivectors_${train_set} \
    --cmvn-opts "--norm-means=true --norm-vars=false" \
    --initial-effective-lrate 0.0017 --final-effective-lrate 0.00017 \
    --cmd "$decode_cmd" \
    --relu-dim 1024 \
    data/${train_set}_hires data/lang $ali_dir $dir  || exit 1;

fi

graph_dir=exp/tri4/graph_sw1_tg
if [ $stage -le 10 ]; then
  for decode_set in train_dev eval2000; do
    (
    num_jobs=`cat data/${decode_set}_hires/utt2spk|cut -d' ' -f2|sort -u|wc -l`
    steps/nnet3/decode.sh --nj $num_jobs --cmd "$decode_cmd" \
        --online-ivector-dir exp/nnet3/ivectors_${decode_set} \
       $graph_dir data/${decode_set}_hires $dir/decode_${decode_set}_hires_sw1_tg || exit 1;
    if $has_fisher; then
	steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
          data/lang_sw1_{tg,fsh_fg} data/${decode_set}_hires \
	  $dir/decode_${decode_set}_hires_sw1_{tg,fsh_fg} || exit 1;
    fi
    ) &
  done
fi
wait;
exit 0;

