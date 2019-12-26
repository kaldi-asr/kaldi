#!/bin/bash

# Copyright 2019 Mobvoi AI Lab, Beijing, China (author: Fangjun Kuang)
# Apache 2.0

set -e

. path.sh


dir=$PWD/exp
mkdir -p $dir

cegs_dir=/cache/fangjun/chain/merged_egs
den_fst_filename=/cache/fangjun/chain/model/den.fst
lda_mat_filename=/cache/fangjun/chain/model/lda.mat

feat_dim=43
output_dim=4464
hidden_dim=625
kernel_size_list="1, 3, 3, 3, 3, 3" # comma separated list
dilation_list="1, 1, 1, 3, 3, 3" # comma separated list

# you may set CUDA_VISIBLE_DEVICES and then set `device_id=0`
device_id=6
num_epochs=4
lr=1e-3

egs_left_context=13
egs_right_context=13


# sort the options alphabetically
python3 ./chain/train.py \
  --device-id $device_id \
  --dilation-list "$dilation_list" \
  --dir $dir \
  --feat-dim $feat_dim \
  --hidden-dim $hidden_dim \
  --kernel-size-list "$kernel_size_list" \
  --lda-mat-filename $lda_mat_filename \
  --output-dim $output_dim \
  --train.cegs-dir $cegs_dir \
  --train.den-fst $den_fst_filename \
  --train.egs-left-context $egs_left_context \
  --train.egs-right-context $egs_right_context \
  --train.lr $lr \
  --train.l2-regularize 5e-4 \
  --train.num-epochs $num_epochs
