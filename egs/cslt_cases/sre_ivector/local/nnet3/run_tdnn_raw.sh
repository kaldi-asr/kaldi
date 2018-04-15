#!/bin/bash

# This is the standard "tdnn" system, built in nnet3 with xconfigs.


set -e -o pipefail -u

stage=0
train_stage=-10
remove_egs=false
srand=0
reporting_email=
common_egs_dir= # use previously dumped egs
dvector_dim=400
vad=false # vad or not when getting egs

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


if [ $# != 3 ]; then
  echo "Usage: $0 [opts] <data-dir> <ali-dir> <exp-dir>"
  exit 1;
fi

train_data_dir=$1
ali_dir=$2
dir=$3

for f in $train_data_dir/feats.scp $ali_dir/{ali.scp,target_num,target_counts}; do
  [ ! -f $f ] && echo "$0: expected file $f to exist" && exit 1
done


###### Bookmark: nnet3 config ######

if [ $stage -le 1 ]; then
  mkdir -p $dir
  echo "$0: creating neural net configs using the xconfig parser";

  num_targets=`cat $ali_dir/target_num`

  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig
  input dim=40 name=input

  # please note that it is important to have input layer with the name=input
  # as the layer immediately preceding the fixed-affine-layer to enable
  # the use of short notation for the descriptor
  fixed-affine-layer name=lda input=Append(-2,-1,0,1,2) affine-transform-file=$dir/configs/lda.mat

  # the first splicing is moved before the lda layer, so no splicing here
  relu-renorm-layer name=tdnn1 dim=650
  relu-renorm-layer name=tdnn2 dim=650 input=Append(-1,0,1)
  relu-renorm-layer name=tdnn3 dim=650 input=Append(-1,0,1)
  relu-renorm-layer name=tdnn4 dim=650 input=Append(-3,0,3)
  relu-renorm-layer name=tdnn5 dim=650 input=Append(-6,-3,0)
  relu-renorm-layer name=tdnn6 dim=$dvector_dim
  output-layer name=output dim=$num_targets max-change=1.5
EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
  echo "num_targets=$num_targets" >> $dir/configs/vars
fi


###### Bookmark: nnet3 training ######

if [ $stage -le 2 ]; then
  steps/nnet3/train_raw_dnn.py --stage=$train_stage \
    --cmd="$cuda_cmd" \
    --feat.cmvn-opts="--norm-means=false --norm-vars=false" \
    --trainer.srand=$srand \
    --trainer.max-param-change=2.0 \
    --trainer.num-epochs=3 \
    --trainer.samples-per-iter=400000 \
    --trainer.optimization.num-jobs-initial=2 \
    --trainer.optimization.num-jobs-final=2 \
    --trainer.optimization.initial-effective-lrate=0.0015 \
    --trainer.optimization.final-effective-lrate=0.00015 \
    --trainer.optimization.minibatch-size=256,128 \
    --egs.dir="$common_egs_dir" \
    --cleanup.remove-egs=$remove_egs \
    --use-gpu=true \
    --reporting.email="$reporting_email" \
    --feat-dir=$train_data_dir \
    --use-dense-targets=false \
    --targets-scp="$ali_dir/ali.scp" \
    --vad-egs=$vad \
    --dir=$dir  || exit 1;
fi


exit 0;
