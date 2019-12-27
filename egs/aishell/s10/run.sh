#!/bin/bash

# Copyright 2019 Mobvoi AI Lab, Beijing, China (author: Fangjun Kuang)
# Apache 2.0

set -e

. path.sh
. cmd.sh

dir=$PWD/exp

stage=0

# checkpoint=./exp/epoch-5.pt

cegs_dir=/cache/fangjun/chain/merged_egs
den_fst_filename=/cache/fangjun/chain/model/den.fst
lda_mat_filename=/cache/fangjun/chain/model/lda.mat

feat_dim=43
output_dim=4464
hidden_dim=625
kernel_size_list="1, 3, 3, 3, 3, 3" # comma separated list
stride_list="1, 1, 3, 1, 1, 1" # comma separated list

# you may set CUDA_VISIBLE_DEVICES and then set `device_id=0`
device_id=7
num_epochs=6
lr=1e-3

egs_left_context=13
egs_right_context=13

log_level=info # valid values: debug, info, warning


# you do NOT need to install tensorflow to use tensorboard
# just use `pip` to install tensorboard
#   to run tensorboard in a server,
#   use `tensorboard --host $server_ip --logdir $exp/tensorboard`

if [[ $stage -le 0 ]]; then
  echo "training..."
  mkdir -p $dir/tensorboard
  # sort the options alphabetically
  python3 ./chain/train.py \
    --checkpoint=${checkpoint:-} \
    --device-id $device_id \
    --dir $dir \
    --feat-dim $feat_dim \
    --hidden-dim $hidden_dim \
    --is-training 1 \
    --kernel-size-list "$kernel_size_list" \
    --lda-mat-filename $lda_mat_filename \
    --log-level $log_level \
    --output-dim $output_dim \
    --stride-list "$stride_list" \
    --train.cegs-dir $cegs_dir \
    --train.den-fst $den_fst_filename \
    --train.egs-left-context $egs_left_context \
    --train.egs-right-context $egs_right_context \
    --train.l2-regularize 5e-4 \
    --train.lr $lr \
    --train.num-epochs $num_epochs
fi

test_feats_scp=/cache/fangjun/chain/test_dataset/feats.scp
model_left_context=12
model_right_context=12

if [[ $stage -le 1 ]]; then
  echo "inference: computing likelihood"
  mkdir -p $dir
  if [[ -f $dir/confidence.scp ]]; then
    echo "$dir/confidence.scp already exist! Skip"
  else
    python3 ./chain/inference.py \
      --checkpoint=${checkpoint:-} \
      --device-id $device_id \
      --dir $dir \
      --feat-dim $feat_dim \
      --feats-scp $test_feats_scp \
      --hidden-dim $hidden_dim \
      --is-training 0 \
      --kernel-size-list "$kernel_size_list" \
      --lda-mat-filename $lda_mat_filename \
      --log-level $log_level \
      --model-left-context $model_left_context \
      --model-right-context $model_right_context \
      --output-dim $output_dim \
      --stride-list "$stride_list"
  fi
fi

graphdir=/cache/fangjun/chain/model/graph
trans_model=/cache/fangjun/chain/model/trans.mdl

if [[ $stage -le 2 ]]; then
  echo "decoding"
  if [[ ! -f $dir/confidence.scp ]]; then
    echo "$dir/confidence.scp does not exist!"
    echo "Please run the inference.py first"
    exit 1
  fi

  ./local/decode.sh \
    $graphdir \
    $trans_model \
    $dir/confidence.scp \
    $dir/decode_res
fi

if [[ $stage -le 3 ]]; then
  echo "scoring"

  ./local/score.sh --cmd "$decode_cmd" \
    /cache/fangjun/chain/test_dataset \
    $graphdir \
    $dir/decode_res || exit 1
fi
