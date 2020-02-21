#!/bin/bash

# Copyright 2020 Mobvoi AI Lab, Beijing, China (author: Fangjun Kuang)
# Apache 2.0

set -e

echo "$0 $@"  # Print the command line for logging

stage=0

device_id=1

train_data_dir=data/train_sp
dev_data_dir=data/dev_sp
test_data_dir=data/test
lang_dir=data/lang

lr=1e-4
num_epochs=6
l2_regularize=1e-5
num_layers=4
hidden_dim=512
proj_dim=200
batch_size=64


dir=exp/ctc

. ./path.sh
. ./cmd.sh

. parse_options.sh

feat_dim=$(feat-to-dim --print-args=false scp:$train_data_dir/feats.scp -)
output_dim=$(cat $lang_dir/phones.list | wc -l)
# added by one since we have an extra blank symbol <blk>
output_dim=$[$output_dim+1]

if [[ $stage -le 0 ]]; then
  mkdir -p $dir

  # sort options alphabetically
  python3 ./ctc/train.py \
    --batch-size $batch_size \
    --device-id $device_id \
    --dir=$dir \
    --feats-scp $train_data_dir/feats.scp \
    --hidden-dim $hidden_dim \
    --input-dim $feat_dim \
    --is-training true \
    --num-layers $num_layers \
    --output-dim $output_dim \
    --proj-dim $proj_dim \
    --train.l2-regularize $l2_regularize \
    --train.labels-scp $train_data_dir/labels.scp \
    --train.lr $lr \
    --train.num-epochs $num_epochs
fi
