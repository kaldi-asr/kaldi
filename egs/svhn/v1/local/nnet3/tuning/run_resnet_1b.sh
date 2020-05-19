#!/usr/bin/env bash

# exp/resnet1b: num-iters=130 nj=2..4 num-params=1.3M dim=96->10 combine=-0.04->-0.04 loglike:train/valid[85,129,final]=(-0.049,-0.044,-0.036/-0.098,-0.085,-0.076) accuracy:train/valid[85,129,final]=(0.9904,0.9908,0.9940/0.9764,0.9804,0.9831)

# This setup is based on the one in cifar/v1/local/nnet3/run_resnet_1{a,b}.sh.
# We are reducing the number of epochs quite a bit, since there is so much
# more data here; and reducing the proportional-shrink value since there is
# more data.
# The augmentation options are changed, with no horizontal flip, less vertical
# shift, and much less horizontal shift.



# Set -e here so that we catch if any executable fails immediately
set -euo pipefail



# training options
stage=0
train_stage=-10
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



dir=exp/resnet${affix}

egs=exp/egs

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

  common="required-time-offsets=0 height-offsets=-1,0,1"
  res_opts="bypass-source=batchnorm"

  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig
  input dim=96 name=input
  conv-layer name=conv1 height-in=32 height-out=32 time-offsets=-1,0,1 required-time-offsets=0 height-offsets=-1,0,1 num-filters-out=$nf1
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
    --trainer.optimization.proportional-shrink=25.0 \
    --trainer.shuffle-buffer-size=2000 \
    --egs.dir="$egs" \
    --use-gpu=true \
    --reporting.email="$reporting_email" \
    --dir=$dir  || exit 1;
fi


exit 0;
