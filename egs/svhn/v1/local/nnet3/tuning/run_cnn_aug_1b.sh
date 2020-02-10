#!/usr/bin/env bash

# run_cnn_aug_1b.sh is like run_cnn_aug_1a.sh but setting
# num-minibatches-history=40.0 (longer history for natural gradient),
# and using the "egs2" examples with more archives, which necessitates
# adjusting the proportional-shrink option (since it should be
# proportional to archive size).

# result improves 97.62 -> 97.71.

# steps/info/nnet3_dir_info.pl exp/cnn_aug1b
# exp/cnn_aug1b: num-iters=180 nj=2..4 num-params=2.8M dim=96->10 combine=-0.06->-0.06 loglike:train/valid[119,179,final]=(-0.066,-0.051,-0.049/-0.126,-0.103,-0.100) accuracy:train/valid[119,179,final]=(0.9846,0.9890,0.9900/0.970,0.9760,0.9771)

# Set -e here so that we catch if any executable fails immediately
set -euo pipefail



# training options
stage=0
train_stage=-10
srand=0
reporting_email=
affix=_aug1b


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



dir=exp/cnn${affix}

egs=exp/egs2

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
  common1="$a required-time-offsets=0 height-offsets=-1,0,1 num-filters-out=40"
  common2="$a required-time-offsets=0 height-offsets=-1,0,1 num-filters-out=80"

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
    --cmd="$cmd" \
    --image.augmentation-opts="--horizontal-shift=0.04 --vertical-shift=0.08 --num-channels=3" \
    --trainer.srand=$srand \
    --trainer.max-param-change=2.0 \
    --trainer.num-epochs=30 \
    --egs.frames-per-eg=1 \
    --trainer.optimization.num-jobs-initial=2 \
    --trainer.optimization.num-jobs-final=4 \
    --trainer.optimization.initial-effective-lrate=0.003 \
    --trainer.optimization.final-effective-lrate=0.0003 \
    --trainer.optimization.minibatch-size=256,128,64 \
    --trainer.optimization.proportional-shrink=18.0 \
    --trainer.shuffle-buffer-size=2000 \
    --egs.dir="$egs" \
    --use-gpu=true \
    --reporting.email="$reporting_email" \
    --dir=$dir  || exit 1;
fi


exit 0;
