#!/bin/bash

# 1d is as 1c but adding rotation in image augmentation.

# local/nnet3/compare.sh exp/resnet1c_cifar10 exp/resnet1d_cifar10
# System                 resnet1c_cifar10   resnet1d_cifar10
# final test accuracy:        0.9514           0.9537      
# final train accuracy:            1           0.9966
# final test objf:         -0.157244          -0.139607
# final train objf:      -0.00751868         -0.0219607
# num-parameters:           1322730            1322730

# local/nnet3/compare.sh exp/resnet1c_cifar100 exp/resnet1d_cifar100
# System                 resnet1c_cifar100 resnet1d_cifar100
# final test accuracy:       0.7627        0.7687
# final train accuracy:       0.96         0.9276
# final test objf:         -0.862205    -0.812203
# final train objf:        -0.174973    -0.265734
# num-parameters:           1345860       1345860
# steps/info/nnet3_dir_info.pl exp/resnet1c_cifar10{,0}
# exp/resnet1d_cifar10: num-iters=133 nj=1..2 num-params=1.3M dim=96->10 combine=-0.04->-0.03 loglike:train/valid[87,132,final]=(-0.153,-0.044,-0.022/-0.25,-0.173,-0.140) accuracy:train/valid[87,132,final]=(0.946,0.9880,0.9966/0.921,0.946,0.954)
# exp/resnet1d_cifar100: num-iters=133 nj=1..2 num-params=1.3M dim=96->100 combine=-0.33->-0.29 loglike:train/valid[87,132,final]=(-0.81,-0.37,-0.27/-1.15,-0.95,-0.81) accuracy:train/valid[87,132,final]=(0.760,0.897,0.928/0.68,0.737,0.769)

# Set -e here so that we catch if any executable fails immediately
set -euo pipefail



# training options
stage=0
train_stage=-10
dataset=cifar10
srand=0
reporting_email=
affix=1d


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



dir=exp/resnet${affix}_${dataset}

egs=exp/${dataset}_egs2

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


  nf1=48
  nf2=96
  nf3=256
  nb3=128

  a="num-minibatches-history=40.0"
  common="$a required-time-offsets=0 height-offsets=-1,0,1"
  res_opts="$a bypass-source=batchnorm"

  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig
  input dim=96 name=input
  conv-layer name=conv1 $a height-in=32 height-out=32 time-offsets=-1,0,1 required-time-offsets=0 height-offsets=-1,0,1 num-filters-out=$nf1
  res-block name=res2 num-filters=$nf1 height=32 time-period=1 $res_opts
  res-block name=res3 num-filters=$nf1 height=32 time-period=1 $res_opts
  conv-layer name=conv4 height-in=32 height-out=16 height-subsample-out=2 time-offsets=-1,0,1 $common num-filters-out=$nf2
  res-block name=res5 num-filters=$nf2 height=16 time-period=2 $res_opts
  res-block name=res6 num-filters=$nf2 height=16 time-period=2 $res_opts
  conv-layer name=conv7 height-in=16 height-out=8 height-subsample-out=2 time-offsets=-2,0,2 $common num-filters-out=$nf3
  res-block name=res8 num-filters=$nf3 num-bottleneck-filters=$nb3 height=8 time-period=4 $res_opts
  res-block name=res9 num-filters=$nf3 num-bottleneck-filters=$nb3 height=8 time-period=4 $res_opts
  res-block name=res10 num-filters=$nf3 num-bottleneck-filters=$nb3 height=8 time-period=4 $res_opts
  channel-average-layer name=channel-average input=Append(2,6,10,14,18,22,24,28) dim=$nf3
  output-layer name=output learning-rate-factor=0.1 dim=$num_targets
EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
fi


if [ $stage -le 2 ]; then

  steps/nnet3/train_raw_dnn.py --stage=$train_stage \
    --cmd="$train_cmd" \
    --image.augmentation-opts="--horizontal-flip-prob=0.5 --horizontal-shift=0.1 --vertical-shift=0.1 --rotation-degree=30 --rotation-prob=0.5 --num-channels=3" \
    --trainer.srand=$srand \
    --trainer.max-param-change=2.0 \
    --trainer.num-epochs=100 \
    --egs.frames-per-eg=1 \
    --trainer.optimization.num-jobs-initial=1 \
    --trainer.optimization.num-jobs-final=2 \
    --trainer.optimization.initial-effective-lrate=0.003 \
    --trainer.optimization.final-effective-lrate=0.0003 \
    --trainer.optimization.minibatch-size=256,128,64 \
    --trainer.optimization.proportional-shrink=50.0 \
    --trainer.shuffle-buffer-size=2000 \
    --egs.dir="$egs" \
    --use-gpu=true \
    --reporting.email="$reporting_email" \
    --dir=$dir  || exit 1;
fi


exit 0;
