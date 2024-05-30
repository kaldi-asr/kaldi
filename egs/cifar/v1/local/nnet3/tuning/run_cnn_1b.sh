#!/usr/bin/env bash


# 1b is like 1a but a smaller model.
# The result is worse.

# steps/info/nnet3_dir_info.pl exp/cnn1b_cifar10
# exp/cnn1b_cifar10: num-iters=60 nj=1..2 num-params=0.1M dim=96->10 combine=-0.39->-0.31 loglike:train/valid[39,59,final]=(-0.55,-0.33,-0.31/-1.00,-1.16,-1.16) accuracy:train/valid[39,59,final]=(0.82,0.89,0.90/0.67,0.66,0.66)



# Set -e here so that we catch if any executable fails immediately
set -euo pipefail



# training options
stage=0
train_stage=-10
dataset=cifar10
srand=0
reporting_email=
affix=1b


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

  common1="required-time-offsets=0 height-offsets=-1,0,1 num-filters-out=16"
  common2="required-time-offsets=0 height-offsets=-1,0,1 num-filters-out=32"

  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig
  input dim=96 name=input
  conv-relu-batchnorm-layer name=cnn1 height-in=32 height-out=32 time-offsets=-1,0,1 $common1
  conv-relu-batchnorm-layer name=cnn2 height-in=32 height-out=32 time-offsets=-1,0,1 $common1
  conv-relu-batchnorm-layer name=cnn3 height-in=32 height-out=32 time-offsets=-1,0,1 $common1
  conv-relu-batchnorm-layer name=cnn4 height-in=32 height-out=16 time-offsets=-1,0,1 $common1 height-subsample-out=2
  conv-relu-batchnorm-layer name=cnn5 height-in=16 height-out=16 time-offsets=-2,0,2 $common1
  conv-relu-batchnorm-layer name=cnn6 height-in=16 height-out=16 time-offsets=-2,0,2 $common1
  conv-relu-batchnorm-layer name=cnn7 height-in=16 height-out=8  time-offsets=-2,0,2 $common1 height-subsample-out=2
  conv-relu-batchnorm-layer name=cnn8 height-in=8 height-out=8   time-offsets=-4,0,4 $common1
  conv-relu-batchnorm-layer name=cnn9 height-in=8 height-out=8   time-offsets=-4,0,4 $common1
  conv-relu-batchnorm-layer name=cnn10 height-in=8 height-out=4   time-offsets=-4,0,4 $common1 height-subsample-out=2
  conv-relu-batchnorm-layer name=cnn11 height-in=4 height-out=4   time-offsets=-8,0,8 $common2
  conv-relu-batchnorm-layer name=cnn12 height-in=4 height-out=4   time-offsets=-8,0,8 $common2
  relu-batchnorm-layer name=fully_connected1 input=Append(0,8,16,24) dim=64
  relu-batchnorm-layer name=fully_connected2 dim=128
  output-layer name=output dim=$num_targets
EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
fi


if [ $stage -le 2 ]; then

  steps/nnet3/train_raw_dnn.py --stage=$train_stage \
    --cmd="$train_cmd" \
    --trainer.srand=$srand \
    --trainer.max-param-change=2.0 \
    --trainer.num-epochs=30 \
    --egs.frames-per-eg=1 \
    --trainer.optimization.num-jobs-initial=1 \
    --trainer.optimization.num-jobs-final=2 \
    --trainer.optimization.initial-effective-lrate=0.0003 \
    --trainer.optimization.final-effective-lrate=0.00003 \
    --trainer.optimization.minibatch-size=256,128,64 \
    --trainer.shuffle-buffer-size=2000 \
    --egs.dir="$egs" \
    --use-gpu=true \
    --reporting.email="$reporting_email" \
    --dir=$dir  || exit 1;
fi


exit 0;
