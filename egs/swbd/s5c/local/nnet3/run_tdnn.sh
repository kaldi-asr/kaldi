#!/bin/bash

# this is the standard "tdnn" system, built in nnet3; it's what we use to
# call multi-splice.

. cmd.sh


# At this script level we don't support not running on GPU, as it would be painfully slow.
# If you want to run without GPU you'd have to call train_tdnn.sh with --gpu false,
# --num-threads 16 and --minibatch-size 128.

stage=0
train_stage=-10
has_fisher=true
speed_perturb=true
use_cnn=true


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

cnn_opts=()
if [ "$use_cnn" == "true" ]; then
# When the cnn option is on, the LDA is replaced by
# an CNN layer at the front of the network
# and the recipe uses fbank features as the CNN input
# As the dimension of the CNN output is usually large, we place
# a linear layer at the output of CNN for dimension reduction
# The ivectors are processed through a fully connected affine layer,
# then concatenated with the CNN bottleneck output and
# passed to the deeper part of the network.
# Due to the data compression issue, it is better to convert MFCC
# to FBANK in the network instead of directly using FBANK features
# from the storage. This script uses MFCC features as its input
# and were converted to FBANK features with an inverse of DCT matrix
# at the first layer of the network.

  cnn_layer="filt_x_dim=3 filt_y_dim=8 filt_x_step=1 filt_y_step=1 num_filters=256 pool_x_size=1 pool_y_size=3 pool_z_size=1 pool_x_step=1 pool_y_step=3 pool_z_step=1"
  cnn_opts+=(--cnn-layer "$cnn_layer")
  cnn_bottleneck_dim=256
  cnn_opts+=(--cnn-bottleneck-dim $cnn_bottleneck_dim)
  cepstral_lifter="22.0"
  cnn_opts+=(--cepstral-lifter "$cepstral_lifter")
  affix=cnn
fi

dir=exp/nnet3/tdnn
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

  steps/nnet3/tdnn/train.sh --stage $train_stage \
    "${cnn_opts[@]}" \
    --num-epochs 2 --num-jobs-initial 3 --num-jobs-final 16 \
    --splice-indexes "-2,-1,0,1,2 -1,2 -3,3 -7,2 0" \
    --feat-type raw \
    --online-ivector-dir exp/nnet3/ivectors_${train_set} \
    --cmvn-opts "--norm-means=false --norm-vars=false" \
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

