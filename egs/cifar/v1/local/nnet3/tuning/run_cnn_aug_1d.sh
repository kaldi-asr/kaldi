#!/usr/bin/env bash


# 1d is as 1c but setting num-minibatches-history=40.
# A bit better.

# local/nnet3/compare.sh exp/cnn_aug_1c_cifar10 exp/cnn_aug_1d_cifar10
# System                cnn_aug_1c_cifar10 cnn_aug_1d_cifar10
# final test accuracy:       0.8834      0.8857
# final train accuracy:       0.9644      0.9626
# final test objf:         -0.362241   -0.356861
# final train objf:        -0.114712   -0.114144
# num-parameters:           2205290     2205290

# local/nnet3/compare.sh exp/cnn_aug_1c_cifar100 exp/cnn_aug_1d_cifar100
# System                cnn_aug_1c_cifar100 cnn_aug_1d_cifar100
# final test accuracy:       0.6219      0.6237
# final train accuracy:       0.8634      0.8688
# final test objf:          -1.42399    -1.40784
# final train objf:        -0.493349   -0.482047
# num-parameters:           2251460     2251460


# steps/info/nnet3_dir_info.pl exp/cnn_aug_1d_cifar10{,0}
# exp/cnn_aug_1d_cifar10: num-iters=200 nj=1..2 num-params=2.2M dim=96->10 combine=-0.24->-0.23 loglike:train/valid[132,199,final]=(-0.172,-0.114,-0.114/-0.38,-0.36,-0.36) accuracy:train/valid[132,199,final]=(0.938,0.963,0.963/0.879,0.887,0.886)
# exp/cnn_aug_1d_cifar100: num-iters=200 nj=1..2 num-params=2.3M dim=96->100 combine=-0.90->-0.92 loglike:train/valid[132,199,final]=(-0.63,-0.48,-0.48/-1.43,-1.41,-1.41) accuracy:train/valid[132,199,final]=(0.821,0.868,0.869/0.61,0.62,0.62)

# Set -e here so that we catch if any executable fails immediately
set -euo pipefail



# training options
stage=0
train_stage=-10
dataset=cifar10
srand=0
reporting_email=
affix=_aug_1d


# End configuration section.
echo "$0 $@"  # Print the command line for logging

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

if ! cuda-compiled; then
  cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi



dir=exp/cnn${affix}_${dataset}

egs=exp/${dataset}_egs

if [ ! -d $egs ]; then
  echo "$0: expected directory $egs to exist.  Run the get_egs.sh commands in the"
  echo "    run.sh before this script."
  exit 1
fi

# check that the expected files are in the egs directory.

for f in $egs/egs.1.ark $egs/train_diagnostic.egs $egs/valid_diagnostic.egs $egs/combine.egs \
         $egs/info/feat_dim $egs/info/left_context $egs/info/right_context \
         $egs/info/output_dim; do
  if [ ! -e $f ]; then
    echo "$0: expected file $f to exist."
    exit 1;
  fi
done


mkdir -p $dir/log


if [ $stage -le 1 ]; then
  mkdir -p $dir
  echo "$0: creating neural net configs using the xconfig parser";

  num_targets=$(cat $egs/info/output_dim)

  # Note: we hardcode in the CNN config that we are dealing with 32x3x color
  # images.

  a="num-minibatches-history=40.0"
  common1="$a required-time-offsets=0 height-offsets=-1,0,1 num-filters-out=32"
  common2="$a required-time-offsets=0 height-offsets=-1,0,1 num-filters-out=64"

  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig
  input dim=96 name=input
  conv-relu-batchnorm-layer name=cnn1 height-in=32 height-out=32 time-offsets=-1,0,1 $common1
  conv-relu-batchnorm-dropout-layer name=cnn2 height-in=32 height-out=16 time-offsets=-1,0,1 dropout-proportion=0.25 $common1 height-subsample-out=2
  conv-relu-batchnorm-layer name=cnn3 height-in=16 height-out=16 time-offsets=-2,0,2 $common2
  conv-relu-batchnorm-dropout-layer name=cnn4 height-in=16 height-out=8 time-offsets=-2,0,2 dropout-proportion=0.25 $common2 height-subsample-out=2
  conv-relu-batchnorm-layer name=cnn5 height-in=8 height-out=8 time-offsets=-4,0,4 $common2
  relu-dropout-layer name=fully_connected1 input=Append(2,6,10,14,18,22,26,30) dropout-proportion=0.5 dim=512
  output-layer name=output dim=$num_targets
EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
fi


if [ $stage -le 2 ]; then

  steps/nnet3/train_raw_dnn.py --stage=$train_stage \
    --cmd="$train_cmd" \
    --image.augmentation-opts="--horizontal-flip-prob=0.5 --horizontal-shift=0.1 --vertical-shift=0.1 --num-channels=3" \
    --trainer.srand=$srand \
    --trainer.max-param-change=2.0 \
    --trainer.num-epochs=100 \
    --egs.frames-per-eg=1 \
    --trainer.optimization.num-jobs-initial=1 \
    --trainer.optimization.num-jobs-final=2 \
    --trainer.optimization.initial-effective-lrate=0.003 \
    --trainer.optimization.final-effective-lrate=0.0001 \
    --trainer.optimization.minibatch-size=256,128,64 \
    --trainer.shuffle-buffer-size=2000 \
    --egs.dir="$egs" \
    --use-gpu=true \
    --reporting.email="$reporting_email" \
    --dir=$dir  || exit 1;
fi


exit 0;
