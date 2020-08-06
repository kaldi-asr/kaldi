#!/bin/bash

# Copyright 2020 Mobvoi AI Lab, Beijing, China (author: Fangjun Kuang)
# Apache 2.0

set -e

echo "$0 $@"  # Print the command line for logging

stage=0
nj=30

export CUDA_VISIBLE_DEVICES="0"
device_id=0

train_data_dir=data/train_sp
dev_data_dir=data/dev_sp
test_data_dir=data/test
lang_dir=data/lang

lr=1e-3
num_epochs=6
l2_regularize=1e-5
batch_size=64

# WARNING(fangjun): You should know how to calculate your
# model's left/right context **manually**
model_left_context=29
model_right_context=29

hidden_dim=1024
bottleneck_dim=128
prefinal_bottleneck_dim=256
kernel_size_list="3, 3, 3, 1, 3, 3, 3, 3, 3, 3, 3, 3" # comma separated list
subsampling_factor_list="1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1" # comma separated list

log_level=info # valid values: debug, info, warning

post_decode_acwt=1

dir=exp/ctc

. ./path.sh
. ./cmd.sh

. parse_options.sh

feat_dim=$(feat-to-dim --print-args=false scp:$train_data_dir/feats.scp -)
output_dim=$(cat $lang_dir/phones.list | wc -l)
# added by one since we have an extra blank symbol <blk>
output_dim=$[$output_dim+1]

pids=()
function kill_trainer() { echo "kill training processes" && kill "${pids[@]}"; }

if [[ $stage -le 0 ]]; then
  mkdir -p $dir/train/tensorboard
  train_checkpoint=
  if [[ -f $dir/train/best_model.pt ]]; then
    train_checkpoint=$dir/train/best_model.pt
  fi

  num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')

  if [[ $num_gpus -gt 1 ]]; then
    echo "$0: training with ddp..."
    echo "$0: number of gpus: $num_gpus"

    export MASTER_ADDR=localhost
    export MASTER_PORT=6666

    for ((i = 0; i < $num_gpus; ++i)); do
      # sort options alphabetically
      python3 ./ctc/ddp_train.py \
        --batch-size $batch_size \
        --checkpoint=${train_checkpoint:-} \
        --device-id $i \
        --dir $dir/train \
        --feats-scp $train_data_dir/feats.scp \
        --hidden-dim $hidden_dim \
        --input-dim $feat_dim \
        --is-training true \
        --model-left-context $model_left_context \
        --model-right-context $model_right_context \
        --num-layers $num_layers \
        --output-dim $output_dim \
        --proj-dim $proj_dim \
        --train.ddp.world-size $num_gpus \
        --train.l2-regularize $l2_regularize \
        --train.labels-scp $train_data_dir/labels.scp \
        --train.lr $lr \
        --train.num-epochs $num_epochs \
        --train.use-ddp true &
      pids+=("$!")
    done
    trap kill_trainer SIGINT SIGTERM
    wait
  else
    echo "$0: training with single gpu..."
    # sort options alphabetically
    python3 ./ctc/train.py \
      --batch-size $batch_size \
      --bottleneck-dim $bottleneck_dim \
      --checkpoint=${train_checkpoint:-} \
      --device-id $device_id \
      --dir $dir/train \
      --feats-scp $train_data_dir/feats.scp \
      --hidden-dim $hidden_dim \
      --input-dim $feat_dim \
      --is-training true \
      --kernel-size-list "$kernel_size_list" \
      --log-level $log_level \
      --model-left-context $model_left_context \
      --model-right-context $model_right_context \
      --output-dim $output_dim \
      --prefinal-bottleneck-dim $prefinal_bottleneck_dim \
      --subsampling-factor-list "$subsampling_factor_list" \
      --train.l2-regularize $l2_regularize \
      --train.labels-scp $train_data_dir/labels.scp \
      --train.lr $lr \
      --train.num-epochs $num_epochs \
      --train.use-ddp false
  fi
fi

if [[ $stage -le 1 ]]; then
  echo "$0: inference: computing likelihood"
  mkdir -p $dir/inference

  for x in $test_data_dir; do
    basename=$(basename $x)
    mkdir -p $dir/inference/$basename
    if [[ -f $dir/inference/$basename/nnet_output.scp ]]; then
      echo "$0: $dir/inference/$basename/nnet_output.scp already exists! Skip"
    else
    best_epoch=$(cat $dir/train/best-epoch-info | grep 'best epoch' | awk '{print $NF}')
    [[ -z $best_epoch ]] && echo "$dir/train/best-epoch-info is not available!" && exit 1
    inference_checkpoint=$dir/train/epoch-${best_epoch}.pt
    echo "$0: using inference checking point: $inference_checkpoint"
    # sort options alphabetically
    python3 ./ctc/inference.py \
      --batch-size $batch_size \
      --bottleneck-dim $bottleneck_dim \
      --checkpoint ${inference_checkpoint:-} \
      --device-id $device_id \
      --dir $dir/inference/$basename \
      --feats-scp $x/feats.scp \
      --hidden-dim $hidden_dim \
      --input-dim $feat_dim \
      --is-training false \
      --kernel-size-list "$kernel_size_list" \
      --log-level $log_level \
      --model-left-context $model_left_context \
      --model-right-context $model_right_context \
      --output-dim $output_dim \
      --prefinal-bottleneck-dim $prefinal_bottleneck_dim \
      --subsampling-factor-list "$subsampling_factor_list"
    fi
  done
fi

if [[ $stage -le 2 ]]; then
  echo "$0: decoding"
  mkdir -p $dir/decode
  for x in $test_data_dir; do
    basename=$(basename $x)
    mkdir -p $dir/decode/$basename

    if [[ ! -f $dir/inference/$basename/nnet_output.scp ]]; then
      echo "$0: $dir/inference/$basename/nnet_output.scp does not exist!"
      echo "$0: Please run inference.py first"
      exit 1
    fi

    echo "$0: decoding $x"

    for i in $(seq $nj); do
      utils/split_scp.pl -j $nj $[$i - 1] $dir/inference/$basename/nnet_output.scp $dir/decode/$basename/nnet_output.$i.scp
    done

    lat_wspecifier="ark:|lattice-scale --acoustic-scale=$post_decode_acwt ark:- ark:- | gzip -c >$dir/decode/$basename/lat.JOB.gz"

    # sort options alphabetically
    $decode_cmd JOB=1:$nj $dir/decode/$basename/log/decode.JOB.log \
      ./local/latgen-faster.py \
      --acoustic-scale=1.0 \
      --allow-partial=true \
      --beam=17.0 \
      --determinize-lattice=false \
      --lattice-beam=8.0 \
      --max-active=7000 \
      --max-mem=200000000 \
      --min-active=200 \
      --minimize=false \
      --word-symbol-table=$lang_dir/words.txt \
      $lang_dir/TLG.fst \
      scp:$dir/decode/$basename/nnet_output.JOB.scp \
      "$lat_wspecifier"
  done
fi

if [[ $stage -le 3 ]]; then
  echo "$0: scoring"

  for x in $test_data_dir; do
    basename=$(basename $x)

    ./local/score.sh --cmd "$decode_cmd" \
      $x \
      $lang_dir \
      $dir/decode/$basename || exit 1
  done

  for x in $test_data_dir; do
    basename=$(basename $x)
    head $dir/decode/$basename/scoring_kaldi/best_*
  done
fi
