#!/bin/bash

# Copyright 2019 Mobvoi AI Lab, Beijing, China (author: Fangjun Kuang)
# Apache 2.0

set -e

stage=0

# GPU device id to use (count from 0).
# you can also set `CUDA_VISIBLE_DEVICES` and set `device_id=0`
device_id=6

nj=10

lang=data/lang_chain # output lang dir
ali_dir=exp/tri5a_ali  # input alignment dir
lat_dir=exp/tri5a_lats # input lat dir
treedir=exp/chain/tri5_tree # output tree dir

# You should know how to calculate your model's left/right context **manually**
model_left_context=28
model_right_context=28
egs_left_context=$[$model_left_context + 1]
egs_right_context=$[$model_right_context + 1]
frames_per_eg=150,110,90
frames_per_iter=1500000
minibatch_size=128

num_epochs=6
lr=1e-3

hidden_dim=1024
bottleneck_dim=128
prefinal_bottleneck_dim=256
kernel_size_list="2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2" # comma separated list
subsampling_factor_list="1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1" # comma separated list

log_level=info # valid values: debug, info, warning

# true to save network output as kaldi::CompressedMatrix
# false to save it as kaldi::Matrix<float>
save_nn_output_as_compressed=false

. ./path.sh
. ./cmd.sh

. parse_options.sh

if [[ $stage -le 0 ]]; then
  for datadir in train dev test; do
    dst_dir=data/mfcc_hires/$datadir
    if [[ ! -f $dst_dir/feats.scp ]]; then
      echo "making mfcc features for LF-MMI training"
      utils/copy_data_dir.sh data/$datadir $dst_dir
      steps/make_mfcc.sh \
        --mfcc-config conf/mfcc_hires.conf \
        --cmd "$train_cmd" \
        --nj $nj \
        $dst_dir || exit 1
      steps/compute_cmvn_stats.sh $dst_dir || exit 1
      utils/fix_data_dir.sh $dst_dir
    else
      echo "$dst_dir/feats.scp already exists."
      echo "kaldi (local/run_tdnn_1b.sh) LF-MMI may have generated it."
      echo "skip $dst_dir"
    fi
  done
fi

if [[ $stage -le 1 ]]; then
  # Create a version of the lang/ directory that has one state per phone in the
  # topo file. [note, it really has two states.. the first one is only repeated
  # once, the second one has zero or more repeats.]
  rm -rf $lang
  cp -r data/lang $lang
  silphonelist=$(cat $lang/phones/silence.csl) || exit 1
  nonsilphonelist=$(cat $lang/phones/nonsilence.csl) || exit 1
  # Use our special topology... note that later on may have to tune this
  # topology.
  steps/nnet3/chain/gen_topo.py $nonsilphonelist $silphonelist >$lang/topo
fi

if [[ $stage -le 2 ]]; then
  # Build a tree using our new topology. This is the critically different
  # step compared with other recipes.
  steps/nnet3/chain/build_tree.sh --frame-subsampling-factor 3 \
      --context-opts "--context-width=2 --central-position=1" \
      --cmd "$train_cmd" 5000 data/mfcc/train $lang $ali_dir $treedir
fi

if  [[ $stage -le 3 ]]; then
  echo "creating phone language-model"
  "$train_cmd" exp/chain/log/make_phone_lm.log \
    chain-est-phone-lm \
     "ark:gunzip -c $treedir/ali.*.gz | ali-to-phones $treedir/final.mdl ark:- ark:- |" \
     exp/chain/phone_lm.fst || exit 1
fi

if [[ $stage -le 4 ]]; then
  echo "creating denominator FST"
  copy-transition-model $treedir/final.mdl exp/chain/0.trans_mdl
  cp $treedir/tree exp/chain
  "$train_cmd" exp/chain/log/make_den_fst.log \
    chain-make-den-fst exp/chain/tree exp/chain/0.trans_mdl exp/chain/phone_lm.fst \
       exp/chain/den.fst exp/chain/normalization.fst || exit 1
fi

if [[ $stage -le 5 ]]; then
  echo "generating egs"
  steps/nnet3/chain/get_egs.sh \
    --alignment-subsampling-factor 3 \
    --cmd "$train_cmd" \
    --cmvn-opts "--norm-means=false --norm-vars=false" \
    --frame-subsampling-factor 3 \
    --frames-overlap-per-eg 0 \
    --frames-per-eg $frames_per_eg \
    --frames-per-iter $frames_per_iter \
    --generate-egs-scp true \
    --left-context $egs_left_context \
    --left-context-initial -1 \
    --left-tolerance 5 \
    --right-context $egs_right_context \
    --right-context-final -1 \
    --right-tolerance 5 \
    --srand 0 \
    --stage -10 \
    data/mfcc_hires/train \
    exp/chain $lat_dir exp/chain/egs
fi

feat_dim=$(cat exp/chain/egs/info/feat_dim)
output_dim=$(cat exp/chain/egs/info/num_pdfs)

if [[ $stage -le 6 ]]; then
  echo "merging egs"
  mkdir -p exp/chain/merged_egs
  num_egs=$(ls -1 exp/chain/egs/cegs*.ark | wc -l)

  run.pl --max-jobs-run $nj JOB=1:$num_egs exp/chain/merged_egs/log/merge_egs.JOB.log \
    nnet3-chain-shuffle-egs ark:exp/chain/egs/cegs.JOB.ark ark:- \| \
    nnet3-chain-merge-egs --minibatch-size=$minibatch_size ark:- \
      ark,scp:exp/chain/merged_egs/cegs.JOB.ark,exp/chain/merged_egs/cegs.JOB.scp || exit 1

  rm exp/chain/egs/cegs.*.ark
fi

if [[ $stage -le 7 ]]; then
  # Note: it might appear that this $lang directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.
  local/mkgraph.sh --self-loop-scale 1.0 data/lang_test exp/chain exp/chain/graph
fi

if [[ $stage -le 8 ]]; then
  echo "training..."

  mkdir -p exp/chain/train/tensorboard
  train_checkpoint=
  if [[ -f exp/chain/train/best_model.pt ]]; then
    train_checkpoint=./exp/chain/train/best_model.pt
  fi

  # sort the options alphabetically
  python3 ./chain/train.py \
    --bottleneck-dim $bottleneck_dim \
    --checkpoint=${train_checkpoint:-} \
    --device-id $device_id \
    --dir exp/chain/train \
    --feat-dim $feat_dim \
    --hidden-dim $hidden_dim \
    --is-training true \
    --kernel-size-list "$kernel_size_list" \
    --log-level $log_level \
    --output-dim $output_dim \
    --prefinal-bottleneck-dim $prefinal_bottleneck_dim \
    --subsampling-factor-list "$subsampling_factor_list" \
    --train.cegs-dir exp/chain/merged_egs \
    --train.den-fst exp/chain/den.fst \
    --train.egs-left-context $egs_left_context \
    --train.egs-right-context $egs_right_context \
    --train.l2-regularize 5e-4 \
    --train.lr $lr \
    --train.num-epochs $num_epochs
fi

if [[ $stage -le 9 ]]; then
  echo "inference: computing likelihood"
  for x in test dev; do
    mkdir -p exp/chain/inference/$x
    if [[ -f exp/chain/inference/$x/nnet_output.scp ]]; then
      echo "exp/chain/inference/$x/nnet_output.scp already exists! Skip"
    else
      best_epoch=$(cat exp/chain/train/best-epoch-info | grep 'best epoch' | awk '{print $NF}')
      inference_checkpoint=exp/chain/train/epoch-${best_epoch}.pt
      python3 ./chain/inference.py \
        --bottleneck-dim $bottleneck_dim \
        --checkpoint $inference_checkpoint \
        --device-id $device_id \
        --dir exp/chain/inference/$x \
        --feat-dim $feat_dim \
        --feats-scp data/mfcc_hires/$x/feats.scp \
        --hidden-dim $hidden_dim \
        --is-training false \
        --kernel-size-list "$kernel_size_list" \
        --log-level $log_level \
        --model-left-context $model_left_context \
        --model-right-context $model_right_context \
        --output-dim $output_dim \
        --prefinal-bottleneck-dim $prefinal_bottleneck_dim \
        --save-as-compressed $save_nn_output_as_compressed \
        --subsampling-factor-list "$subsampling_factor_list" || exit 1
    fi
  done
fi

if [[ $stage -le 10 ]]; then
  echo "decoding"
  for x in test dev; do
    if [[ ! -f exp/chain/inference/$x/nnet_output.scp ]]; then
      echo "exp/chain/inference/$x/nnet_output.scp does not exist!"
      echo "Please run inference.py first"
      exit 1
    fi
    echo "decoding $x"

    ./local/decode.sh \
      --nj $nj \
      exp/chain/graph \
      exp/chain/0.trans_mdl \
      exp/chain/inference/$x/nnet_output.scp \
      exp/chain/decode_res/$x
  done
fi

if [[ $stage -le 11 ]]; then
  echo "scoring"

  for x in test dev; do
    ./local/score.sh --cmd "$decode_cmd" \
      data/mfcc_hires/$x \
      exp/chain/graph \
      exp/chain/decode_res/$x || exit 1
  done

  for x in test dev; do
    head exp/chain/decode_res/$x/scoring_kaldi/best_*
  done
fi
