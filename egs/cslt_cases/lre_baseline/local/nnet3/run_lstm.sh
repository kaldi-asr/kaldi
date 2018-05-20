#!/bin/bash

# This is the standard "lstm" system, built in nnet3 with xconfigs.


set -e -o pipefail -u

stage=0
train_stage=-10
remove_egs=false
srand=0
reporting_email=
common_egs_dir= # use previously dumped egs
vad=false # vad or not when getting egs

# LSTM options
train_stage=-10
label_delay=5

# training chunk-options
chunk_width=40,30,20
chunk_left_context=40
chunk_right_context=0


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
  fixed-affine-layer name=lda input=Append(-2,-1,0,1,2) affine-transform-file=$dir/configs/lda.mat delay=$label_delay

  fast-lstmp-layer name=lstm cell-dim=512 recurrent-projection-dim=256 non-recurrent-projection-dim=256
  output-layer name=output output-delay=$label_delay dim=$num_targets max-change=1.5
EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
  echo "num_targets=$num_targets" >> $dir/configs/vars
fi


###### Bookmark: nnet3 training ######

if [ $stage -le 2 ]; then
  steps/nnet3/train_raw_rnn.py --stage=$train_stage \
    --cmd="$cuda_cmd" \
    --feat.cmvn-opts="--norm-means=false --norm-vars=false" \
    --trainer.srand=$srand \
    --trainer.max-param-change=2.0 \
    --trainer.num-epochs=6 \
    --trainer.deriv-truncate-margin=10 \
    --trainer.samples-per-iter=20000 \
    --trainer.optimization.num-jobs-initial=2 \
    --trainer.optimization.num-jobs-final=8 \
    --trainer.optimization.initial-effective-lrate=0.0003 \
    --trainer.optimization.final-effective-lrate=0.00003 \
    --trainer.optimization.shrink-value 0.99 \
    --trainer.rnn.num-chunk-per-minibatch=128,64 \
    --trainer.optimization.momentum=0.5 \
    --egs.chunk-width=$chunk_width \
    --egs.chunk-left-context=$chunk_left_context \
    --egs.chunk-right-context=$chunk_right_context \
    --egs.chunk-left-context-initial=0 \
    --egs.chunk-right-context-final=0 \
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
