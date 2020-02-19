#!/bin/bash

# Copyright 2019 Mobvoi AI Lab, Beijing, China (author: Fangjun Kuang)
# Apache 2.0

set -e

stage=0

export CUDA_VISIBLE_DEVICES="0,3"
device_id=0

nj=10

train_set=train_cleaned
gmm_dir=exp/tri3_cleaned

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
kernel_size_list="3, 3, 3, 1, 3, 3, 3, 3, 3, 3, 3, 3" # comma separated list
subsampling_factor_list="1, 1, 1, 3, 1, 1, 1, 1, 1, 1, 1, 1" # comma separated list

log_level=info # valid values: debug, info, warning

# true to save network output as kaldi::CompressedMatrix
# false to save it as kaldi::Matrix<float>
save_nn_output_as_compressed=false

. ./path.sh
. ./cmd.sh

. parse_options.sh

ali_dir=${gmm_dir}_ali_${train_set}_sp  # output ali dir
lat_dir=${gmm_dir}_lat_${train_set}_sp  # output lat dir
tree_dir=${gmm_dir}_tree_${train_set}_sp  # output tree dir
train_data_dir=data/${train_set}_sp_hires
lores_train_data_dir=data/${train_set}_sp

if [[ $stage -le 0 ]]; then
  echo "$0: preparing directory for low-resolution speed-perturbed data (for alignment)"
  utils/data/perturb_data_dir_speed_3way.sh data/$train_set data/${train_set}_sp

  for x in ${train_set}_sp dev test; do
    utils/copy_data_dir.sh data/$x data/${x}_hires
  done
fi

if [[ $stage -le 1 ]]; then
  echo "$0: making MFCC features for low-resolution speed-perturbed data"
  steps/make_mfcc.sh --nj $nj --cmd "$train_cmd" data/${train_set}_sp
  steps/compute_cmvn_stats.sh data/${train_set}_sp
  echo "fixing input data-dir to remove nonexistent features, in case some "
  echo ".. speed-perturbed segments were too short."
  utils/fix_data_dir.sh data/${train_set}_sp
fi

if [[ $stage -le 2 ]]; then
  echo "$0: aligning with the perturbed low-resolution data"
  steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" \
    data/${train_set}_sp data/lang $gmm_dir $ali_dir
fi

if [[ $stage -le 3 ]]; then
  echo "$0: creating high-resolution MFCC features"

  # do volume-perturbation on the training data prior to extracting hires
  # features; this helps make trained nnets more invariant to test data volume.
  utils/data/perturb_data_dir_volume.sh data/${train_set}_sp_hires

  for x in ${train_set}_sp dev test; do
    steps/make_mfcc.sh --nj $nj --mfcc-config conf/mfcc_hires.conf \
      --cmd "$train_cmd" data/${x}_hires
    steps/compute_cmvn_stats.sh data/${x}_hires
    utils/fix_data_dir.sh data/${x}_hires
  done
fi

if [[ $stage -le 4 ]]; then
  for f in $gmm_dir/final.mdl $train_data_dir/feats.scp \
      $lores_train_data_dir/feats.scp $ali_dir/ali.1.gz $gmm_dir/final.mdl; do
    [ ! -f $f ] && echo "$0: expected file $f to exist" && exit 1
  done
fi

if [[ $stage -le 5 ]]; then
  echo "$0: creating lang directory with one state per phone."
  # Create a version of the lang/ directory that has one state per phone in the
  # topo file. [note, it really has two states.. the first one is only repeated
  # once, the second one has zero or more repeats.]
  cp -r data/lang data/lang_chain
  silphonelist=$(cat data/lang_chain/phones/silence.csl) || exit 1;
  nonsilphonelist=$(cat data/lang_chain/phones/nonsilence.csl) || exit 1;
  # Use our special topology... note that later on may have to tune this
  # topology.
  steps/nnet3/chain/gen_topo.py $nonsilphonelist $silphonelist >data/lang_chain/topo
fi

if [[ $stage -le 6 ]]; then
  # Get the alignments as lattices (gives the chain training more freedom).
  # use the same num-jobs as the alignments
  steps/align_fmllr_lats.sh --nj $nj --cmd "$train_cmd" ${lores_train_data_dir} \
    data/lang $gmm_dir $lat_dir
  rm $lat_dir/fsts.*.gz # save space
fi

if [[ $stage -le 7 ]]; then
  # Build a tree using our new topology.  We know we have alignments for the
  # speed-perturbed data (local/nnet3/run_ivector_common.sh made them), so use
  # those.
  steps/nnet3/chain/build_tree.sh --frame-subsampling-factor 3 \
      --context-opts "--context-width=2 --central-position=1" \
      --cmd "$train_cmd" 4000 ${lores_train_data_dir} data/lang_chain $ali_dir $tree_dir
fi

if  [[ $stage -le 8 ]]; then
  echo "$0: creating phone language-model"
  "$train_cmd" exp/chain/log/make_phone_lm.log \
    chain-est-phone-lm \
     "ark:gunzip -c $tree_dir/ali.*.gz | ali-to-phones $tree_dir/final.mdl ark:- ark:- |" \
     exp/chain/phone_lm.fst || exit 1
fi

if [[ $stage -le 9 ]]; then
  echo "creating denominator FST"
  copy-transition-model $tree_dir/final.mdl exp/chain/0.trans_mdl
  cp $tree_dir/tree exp/chain
  "$train_cmd" exp/chain/log/make_den_fst.log \
    chain-make-den-fst exp/chain/tree exp/chain/0.trans_mdl exp/chain/phone_lm.fst \
       exp/chain/den.fst exp/chain/normalization.fst || exit 1
fi

if [[ $stage -le 10 ]]; then
  echo "$0: generating egs"
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
    $train_data_dir \
    exp/chain $lat_dir exp/chain/egs
fi

feat_dim=$(cat exp/chain/egs/info/feat_dim)
output_dim=$(cat exp/chain/egs/info/num_pdfs)

if [[ $stage -le 11 ]]; then
  echo "$0: merging egs"
  mkdir -p exp/chain/merged_egs
  num_egs=$(ls -1 exp/chain/egs/cegs*.ark | wc -l)

  run.pl --max-jobs-run $nj JOB=1:$num_egs exp/chain/merged_egs/log/merge_egs.JOB.log \
    nnet3-chain-shuffle-egs ark:exp/chain/egs/cegs.JOB.ark ark:- \| \
    nnet3-chain-merge-egs --minibatch-size=$minibatch_size ark:- \
      ark,scp:exp/chain/merged_egs/cegs.JOB.ark,exp/chain/merged_egs/cegs.JOB.scp || exit 1

  rm exp/chain/egs/cegs.*.ark
fi

if [[ $stage -le 12 ]]; then
  # Note: it might appear that this $lang directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.
  local/mkgraph.sh --self-loop-scale 1.0 data/lang_test exp/chain exp/chain/graph
fi

if [[ $stage -le 13 ]]; then
  echo "$0: training..."

  mkdir -p exp/chain/train/tensorboard
  train_checkpoint=
  if [[ -f exp/chain/train/best_model.pt ]]; then
    train_checkpoint=./exp/chain/train/best_model.pt
  fi

  num_gpus=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')

  if [[ $num_gpus -gt 1 ]]; then
    echo "$0: training with ddp..."

    export MASTER_ADDR=localhost
    export MASTER_PORT=6666

    for ((i = 0; i < $num_gpus; ++i)); do
      # sort the options alphabetically
      python3 ./chain/ddp_train.py \
        --bottleneck-dim $bottleneck_dim \
        --checkpoint=${train_checkpoint:-} \
        --device-id $i \
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
        --train.ddp.world-size $num_gpus \
        --train.den-fst exp/chain/den.fst \
        --train.egs-left-context $egs_left_context \
        --train.egs-right-context $egs_right_context \
        --train.l2-regularize 5e-5 \
        --train.leaky-hmm-coefficient 0.1 \
        --train.lr $lr \
        --train.num-epochs $num_epochs \
        --train.use-ddp true \
        --train.valid-cegs-scp exp/chain/egs/valid_diagnostic.scp \
        --train.xent-regularize 0.1 &
    done
    wait
  else
    echo "$0: training with single gpu..."
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
      --train.l2-regularize 5e-5 \
      --train.leaky-hmm-coefficient 0.1 \
      --train.lr $lr \
      --train.num-epochs $num_epochs \
      --train.use-ddp false \
      --train.valid-cegs-scp exp/chain/egs/valid_diagnostic.scp \
      --train.xent-regularize 0.1
  fi
fi

if [[ $stage -le 14 ]]; then
  echo "$0: inference: computing likelihood"
  for x in test dev; do
    mkdir -p exp/chain/inference/${x}_hires
    if [[ -f exp/chain/inference/${x}_hires/nnet_output.scp ]]; then
      echo "$0: exp/chain/inference/${x}_hires/nnet_output.scp already exists! Skip"
    else
      best_epoch=$(cat exp/chain/train/best-epoch-info | grep 'best epoch' | awk '{print $NF}')
      inference_checkpoint=exp/chain/train/epoch-${best_epoch}.pt
      python3 ./chain/inference.py \
        --bottleneck-dim $bottleneck_dim \
        --checkpoint $inference_checkpoint \
        --device-id $device_id \
        --dir exp/chain/inference/${x}_hires \
        --feat-dim $feat_dim \
        --feats-scp data/${x}_hires/feats.scp \
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

if [[ $stage -le 15 ]]; then
  echo "$0: decoding"
  for x in test dev; do
    if [[ ! -f exp/chain/inference/${x}_hires/nnet_output.scp ]]; then
      echo "$0: exp/chain/inference/${x}_hires/nnet_output.scp does not exist!"
      echo "$0: Please run inference.py first"
      exit 1
    fi
    echo "$0: decoding ${x}_hires"

    ./local/decode.sh \
      --nj $nj \
      exp/chain/graph \
      exp/chain/0.trans_mdl \
      exp/chain/inference/${x}_hires/nnet_output.scp \
      exp/chain/decode_res/${x}_hires
  done
fi

if [[ $stage -le 16 ]]; then
  echo "$0: scoring"

  for x in test dev; do
    ./local/score.sh --cmd "$decode_cmd" \
      data/${x}_hires \
      exp/chain/graph \
      exp/chain/decode_res/${x}_hires || exit 1
  done

  for x in test dev; do
    head exp/chain/decode_res/${x}_hires/scoring_kaldi/best_*
  done
fi
