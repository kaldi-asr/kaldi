#!/bin/bash

# 1b46 is as 1b46 but changing the --output-epsilon to 0.2.
# 1b45 is as 1b43 but changing the --output-epsilon to 0.1.
# 1b43 is as 1b42 but changing the --output-epsilon to 0.01.

# This technique doesn't sem to be helpful.
# local/nnet3/compare.sh exp/resnet1b32_cifar10 exp/resnet1b46_cifar10
# System                resnet1b32_cifar10 resnet1b46_cifar10
# final test accuracy:       0.9095      0.9062
# final train accuracy:        0.998      0.9888
# final test objf:         -0.327783   -0.365697
# final train objf:       -0.0139392   -0.162269
# num-parameters:            752010      752010

# local/nnet3/compare.sh exp/resnet1b32_cifar100 exp/resnet1b46_cifar100
# System                resnet1b32_cifar100 resnet1b46_cifar100
# final test accuracy:       0.6676      0.6665
# final train accuracy:       0.9174      0.8988
# final test objf:          -1.27745    -1.28075
# final train objf:        -0.291935   -0.550797
# num-parameters:            775140      775140


# 1b42 is as 1b32 but adding --output-epsilon=0.0001 to the
# augmentation opts.  This will tend to prevent the probabilities
# for the wrong classes from getting too small, which acts
# a bit like regularization.


# 1b32 is as 1b27 but using the "egs2" egs, with more data per archive.
# This is as a more relevant baseline for 30 and 31.
# It's about the same (maybe 0.15% or 0.2% worse).

# local/nnet3/compare.sh exp/resnet1b27_cifar10 exp/resnet1b32_cifar10
# System                resnet1b27_cifar10 resnet1b32_cifar10
# final test accuracy:       0.9109      0.9095
# final train accuracy:       0.9968       0.998
# final test objf:         -0.341826   -0.327783
# final train objf:       -0.0166946  -0.0139392
# num-parameters:            752010      752010

# local/nnet3/compare.sh exp/resnet1b27_cifar100 exp/resnet1b32_cifar100
# System                resnet1b27_cifar100 resnet1b32_cifar100
# final test accuracy:       0.6696      0.6676
# final train accuracy:       0.9234      0.9174
# final test objf:          -1.28065    -1.27745
# final train objf:        -0.284288   -0.291935
# num-parameters:            775140      775140


#
# 1b27 is as 1b25 but reducing the bottleneck dimension from 128 to 96.

# 1b25 is as 1b24 but using a larger number (256, not 128) of
# filters near the end, with a bottleneck in the res-block layers of
# 128 filters; and having 3 of these layers instead of 2.
# actually num-params is increased 0.87 -> 0.99 million.
#
# valid accuracy 1b19=0.9114, 1b24=0.9109, 1b25=0.9154.
# train accuracy 1b19=0.9964, 1b24=0.9978, 1b25=0.9988.
#
#

# 1b24 is as 1b19 but introducing a channel-average-layer
# to do channel averaging instead of appending stuff.  Also
# increasing $nf3 to compensate for the reduced parameters

# 1b19 is as 1b7 but after changing convolution.py to generate
# the res-block config in a different way, without a
# redundant convolution layer.

# 1b7 is as 1b6 but reducing the dim of the relu layer at the end from
#   256 to nf3=96, to reduce params.
#     The final test accuracy is slightly better than 1b6, 0.9095 -> 0.9111.

# 1b6 is as 1b4 but reducing nf3 from 128 to 96, to reduce the num-params.
#   See also 1b5 where we reduced the num-params by removing a layer.
# 1b4 is as 1b3 but the same change as 1f->1e, removing the last convolutional layer

# 1b3 is as 1b2 but changing direct-convolution-source to batchnorm.
# 1b2 is as 1b but re-running after fixing a bug in image augmentation.
# 1b is as 1a but adding batchnorm before relu at last layer, and removing the dropout.
# also reverting the learning rates to be like our normal learning rates (10x larger
# than 1a).

# run_resnet_1a.sh is modified run_cnn_aug_1b.sh (i.e. it has augmentation), but
# it's very unlike the baseline- we are moving to a resnet-like architecture.
# (however, when changing num-filters or when downsampling, we use a regular
# convolutional layer).
#


#

# Set -e here so that we catch if any executable fails immediately
set -euo pipefail



# training options
stage=0
train_stage=-10
dataset=cifar10
srand=0
reporting_email=
affix=1b46


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


  nf1=32
  nf2=64
  nf3=256
  nb3=96

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
    --cmd="$train_cmd" \
    --image.augmentation-opts="--horizontal-flip-prob=0.5 --horizontal-shift=0.1 --vertical-shift=0.1 --num-channels=3 --output-epsilon=0.2" \
    --trainer.srand=$srand \
    --trainer.max-param-change=2.0 \
    --trainer.num-epochs=30 \
    --egs.frames-per-eg=1 \
    --trainer.optimization.num-jobs-initial=1 \
    --trainer.optimization.num-jobs-final=2 \
    --trainer.optimization.initial-effective-lrate=0.003 \
    --trainer.optimization.final-effective-lrate=0.0003 \
    --trainer.optimization.minibatch-size=256,128,64 \
    --trainer.shuffle-buffer-size=2000 \
    --egs.dir="$egs" \
    --use-gpu=true \
    --reporting.email="$reporting_email" \
    --dir=$dir  || exit 1;
fi


exit 0;
