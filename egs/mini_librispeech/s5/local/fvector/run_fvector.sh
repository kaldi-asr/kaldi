#!/bin/bash

. ./cmd.sh
set -e

stage=3
train_stage=-10
data=data/train_clean_5
noise_data=data/noise
egs_dir=exp/fvector/egs
fvector_dir=exp/fvector
use_gpu=true

. ./path.sh
. ./cmd.sh
. ./utils/parse_options.sh

if [ $stage -le 3 ]; then
  #dump egs                                         
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $egs_dir/storage ]; then  
    utils/create_split_dir.pl \
    /export/b{11,12,13}/$USER/kaldi-data/egs/minilibrispeech-$(date +'%m_%d_%H_%M')/s5/$egs_dir/storage $egs_dir/storage
  fi

  steps/nnet3/fvector/get_egs.sh --cmd "$train_cmd" \
    --nj 8 \
    --stage 0 \
    --egs-per-iter 12500 \
    --egs-per-iter-diagnostic 10000 \
    --num-diagnostic-percent 5 \
    --frame-length 25 \
    --left-padding 1 \
    --right-padding 1 \
    "$data" "$noise_data" "$egs_dir"
fi

if [ $stage -le 4 ]; then
  #prepare configs
  echo "$0: creating neural net configs using the xconfig parser";
  #options
  input_dim=400
  num_filters=100

  mkdir -p $fvector_dir/configs

  cat <<EOF > $fvector_dir/configs/network.xconfig
  input dim=$input_dim name=input
  # Each eg contains 8 frames, do Frequency-domain feature learning, and then
  # use TDNN model split it into one vector
  preprocess-fft-abs-lognorm-affine-log-layer name=raw0 cos-transform-file=$fvector_dir/configs/cos_transform.mat sin-transform-file=$fvector_dir/configs/sin_transform.mat num-filters=$num_filters half-fft-range=true
  conv-relu-batchnorm-layer name=cnn1 height-in=$num_filters height-out=$[$num_filters/2] time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=64 height-subsample-out=2 learning-rate-factor=0.34 max-change=0.25

  relu-batchnorm-layer name=tdnn0 input=cnn1 dim=625
  relu-batchnorm-layer name=tdnn1 input=Append(0,1,2) dim=625
  relu-batchnorm-layer name=tdnn2 input=Append(0,1,2) dim=625
  relu-batchnorm-layer name=tdnn3 input=Append(0,1,2) dim=625
  relu-batchnorm-layer name=tdnn4 input=Append(0,1) dim=625
  output-layer name=output input=tdnn4 dim=200 include-log-softmax=False param-stddev=0.04 bias-stddev=1.0
EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $fvector_dir/configs/network.xconfig --config-dir $fvector_dir/configs/
  # Modify the final.config and generate sin.mat/cos.mat manually
  python local/fvector/add_output_node.py --input-dim 400 --output-dim 200 --config-file $fvector_dir/configs/final.config
  python local/fvector/generate_sin_cos_matrix.py \
    --feat-dim 400 --dir $fvector_dir
fi

if [ $stage -le 5 ]; then
  #training
  steps/nnet3/xvector/train.sh --cmd "$train_cmd" \
    --initial-effective-lrate 0.002 \
    --final-effective-lrate 0.0002 \
    --max-param-change 0.2 \
    --minibatch-size 16 \
    --num-epochs 8 --use-gpu $use_gpu --stage $train_stage \
    --num-jobs-initial 1 --num-jobs-final 5 \
    --egs-dir $egs_dir \
    $fvector_dir
fi
