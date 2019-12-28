#!/bin/bash

# Copyright 2019 Mobvoi AI Lab, Beijing, China (author: Fangjun Kuang)
# Apache 2.0

# This file demonstrates how to run LF-MMI training in PyTorch
# with kaldi pybind. The neural network part is based on PyTorch,
# while all other parts are based on kaldi.
#
# We assume that you have built kaldi pybind and installed PyTorch.
# You also need a GPU to run this example.
#
# PyTorch with version `1.3.0dev20191006` has been tested and is
# guaranteed to work.
#
# Note that we have used Tensorboard to visualize the training loss.
# You do **NOT** need to install TensorFlow to use Tensorboard

. ./cmd.sh
. ./path.sh

data=/data/fangjunkuang/data/aishell
data_url=www.openslr.org/resources/33

nj=30

# GPU device id to use (count from 0).
# you can also set `CUDA_VISIBLE_DEVICES` and set `device_id=0`
device_id=7

stage=0

if [[ $stage -le 0 ]]; then
  local/download_and_untar.sh $data $data_url data_aishell || exit 1
  local/download_and_untar.sh $data $data_url resource_aishell || exit 1
fi

if [[ $stage -le 1 ]]; then
  local/aishell_prepare_dict.sh $data/resource_aishell || exit 1
  # generated in data/local/dict
fi

if [[ $stage -le 2 ]]; then
  local/aishell_data_prep.sh $data/data_aishell/wav \
    $data/data_aishell/transcript || exit 1
  # generated in data/{train,test,dev}/{spk2utt text utt2spk wav.scp}
fi

if [[ $stage -le 3 ]]; then
  utils/prepare_lang.sh --position-dependent-phones false data/local/dict \
      "<SPOKEN_NOISE>" data/local/lang data/lang || exit 1
fi

if [[ $stage -le 4 ]]; then
  local/aishell_train_lms.sh || exit 1
  utils/format_lm.sh data/lang data/local/lm/3gram-mincount/lm_unpruned.gz \
    data/local/dict/lexicon.txt data/lang_test || exit 1
  cp data/lang/phones/* data/lang_test/phones/
fi

mfccdir=mfcc
if [[ $stage -le 5 ]]; then
  for x in train dev test; do
    steps/make_mfcc_pitch.sh --cmd $train_cmd --nj $nj \
      data/$x exp/make_mfcc/$x $mfccdir || exit 1
    steps/compute_cmvn_stats.sh data/$x exp/make_mfcc/$x $mfccdir || exit 1
    utils/fix_data_dir.sh data/$x || exit 1
  done
fi

if [[ $stage -le 6 ]]; then
  steps/train_mono.sh --cmd $train_cmd --nj $nj \
    data/train data/lang exp/mono || exit 1
fi

if [[ $stage -le 7 ]]; then
  steps/align_si.sh --cmd $train_cmd --nj $nj \
    data/train data/lang exp/mono exp/mono_ali || exit 1
fi

if [[ $stage -le 8 ]]; then
  steps/train_deltas.sh --cmd $train_cmd \
   2500 20000 data/train data/lang exp/mono_ali exp/tri1 || exit 1
fi

if [[ $stage -le 9 ]]; then
  steps/align_si.sh --cmd $train_cmd --nj $nj \
    data/train data/lang exp/tri1 exp/tri1_ali || exit 1
fi

if [[ $stage -le 10 ]]; then
  steps/train_deltas.sh --cmd $train_cmd \
   2500 20000 data/train data/lang exp/tri1_ali exp/tri2 || exit 1
fi

if [[ $stage -le 11 ]]; then
  steps/align_si.sh --cmd $train_cmd --nj $nj \
    data/train data/lang exp/tri2 exp/tri2_ali || exit 1
fi

if [[ $stage -le 12 ]]; then
  steps/train_lda_mllt.sh --cmd $train_cmd \
   2500 20000 data/train data/lang exp/tri2_ali exp/tri3a || exit 1
fi

if [[ $stage -le 13 ]]; then
  steps/align_fmllr.sh --cmd $train_cmd --nj $nj \
    data/train data/lang exp/tri3a exp/tri3a_ali || exit 1
fi

if [[ $stage -le 14 ]]; then
  steps/align_fmllr_lats.sh --nj $nj --cmd $train_cmd data/train \
    data/lang exp/tri3a exp/tri3a_lats
  rm exp/tri3a_lats/fsts.*.gz # save space
fi

if [[ $stage -le 15 ]]; then
  for datadir in train dev test; do
    dst_dir=data/fbank_pitch/$datadir
    utils/copy_data_dir.sh data/$datadir $dst_dir
    echo "$0: making bank-pitches features for LF-MMI training"
    steps/make_fbank_pitch.sh --cmd $train_cmd --nj $nj $dst_dir || exit 1
    steps/compute_cmvn_stats.sh $dst_dir || exit 1
    utils/fix_data_dir.sh $dst_dir
  done
fi

lang=data/lang_chain

if [[ $stage -le 16 ]]; then
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

ali_dir=exp/tri3a_ali
treedir=exp/chain/tri3_tree
if [[ $stage -le 17 ]]; then
  # Build a tree using our new topology. This is the critically different
  # step compared with other recipes.
  steps/nnet3/chain/build_tree.sh --frame-subsampling-factor 3 \
      --context-opts "--context-width=2 --central-position=1" \
      --cmd $train_cmd 5000 data/train $lang $ali_dir $treedir
fi

if  [[ $stage -le 18 ]]; then
  echo "creating phone language-model"
  $train_cmd exp/chain/log/make_phone_lm.log \
    chain-est-phone-lm \
     "ark:gunzip -c $treedir/ali.*.gz | ali-to-phones $treedir/final.mdl ark:- ark:- |" \
     exp/chain/phone_lm.fst || exit 1
fi

if [[ $stage -le 19 ]]; then
  echo "creating denominator FST"
  copy-transition-model $treedir/final.mdl exp/chain/0.trans_mdl
  cp $treedir/tree exp/chain
  $train_cmd exp/chain/log/make_den_fst.log \
    chain-make-den-fst exp/chain/tree exp/chain/0.trans_mdl exp/chain/phone_lm.fst \
       exp/chain/den.fst exp/chain/normalization.fst || exit 1
fi

# You should know how to calculate your model's left/right context **manually**
model_left_context=12
model_right_context=12
egs_left_context=$[$model_left_context + 1]
egs_right_context=$[$model_right_context + 1]
frames_per_eg=150,110,90

if [[ $stage -le 20 ]]; then
  echo "generating egs"
  steps/nnet3/chain/get_egs.sh \
    --alignment-subsampling-factor 3 \
    --cmd $train_cmd \
    --cmvn-opts "--norm-means=false --norm-vars=false" \
    --frame-subsampling-factor 3 \
    --frames-overlap-per-eg 0 \
    --frames-per-eg $frames_per_eg \
    --frames-per-iter 1500000 \
    --generate-egs-scp true \
    --left-context $egs_left_context \
    --left-context-initial -1 \
    --left-tolerance 5 \
    --right-context $egs_right_context \
    --right-context-final -1 \
    --right-tolerance 5 \
    --srand 0 \
    --stage -10 \
    data/fbank_pitch/train \
    exp/chain exp/tri3a_lats exp/chain/egs
fi

if [[ $stage -le 21 ]]; then
  echo "merging egs"
  mkdir -p exp/chain/merged_egs
  num_egs=$(ls -1 exp/chain/egs/cegs*.ark | wc -l)

  minibatch_size=128

  run.pl --max-jobs-run $nj JOB=1:$num_egs exp/chain/merged_egs/log/merge_egs.JOB.log \
    nnet3-chain-shuffle-egs ark:exp/chain/egs/cegs.JOB.ark ark:- \| \
    nnet3-chain-merge-egs --minibatch-size=$minibatch_size ark:- \
      ark,scp:exp/chain/merged_egs/cegs.JOB.ark,exp/chain/merged_egs/cegs.JOB.scp || exit 1

  rm exp/chain/egs/cegs.*.ark
fi

if [[ $stage -le 22 ]]; then
  # Note: it might appear that this $lang directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.
  local/mkgraph.sh --self-loop-scale 1.0 data/lang_test exp/chain exp/chain/graph
fi

num_epochs=10
lr=2e-3

feat_dim=$(cat exp/chain/egs/info/feat_dim)
output_dim=$(cat exp/chain/egs/info/num_pdfs)

hidden_dim=625
kernel_size_list="1, 3, 3, 3, 3, 3" # comma separated list
stride_list="1, 1, 3, 1, 1, 1" # comma separated list

log_level=info # valid values: debug, info, warning

if [[ $stage -le 24 ]]; then
  echo "training..."

  mkdir -p exp/chain/train/tensorboard
  # train_checkpoint=./exp/exp/train/epoch-5.pt
  # sort the options alphabetically
  python3 ./chain/train.py \
    --checkpoint=${train_checkpoint:-} \
    --device-id $device_id \
    --dir exp/chain/train \
    --feat-dim $feat_dim \
    --hidden-dim $hidden_dim \
    --is-training 1 \
    --kernel-size-list "$kernel_size_list" \
    --log-level $log_level \
    --output-dim $output_dim \
    --stride-list "$stride_list" \
    --train.cegs-dir exp/chain/merged_egs \
    --train.den-fst exp/chain/den.fst \
    --train.egs-left-context $egs_left_context \
    --train.egs-right-context $egs_right_context \
    --train.l2-regularize 5e-4 \
    --train.lr $lr \
    --train.num-epochs $num_epochs
fi

best_epoch=$(cat exp/chain/train/best-epoch-info | grep 'best epoch' | awk '{print $NF}')
inference_checkpoint=exp/chain/train/epoch-${best_epoch}.pt

if [[ $stage -le 25 ]]; then
  echo "inference: computing likelihood"
  for x in test dev; do
    mkdir -p exp/chain/inference/$x
    if [[ -f exp/chain/inference/$x/confidence.scp ]]; then
      echo "exp/chain/inference/$x/confidence.scp already exists! Skip"
    else
      python3 ./chain/inference.py \
        --checkpoint $inference_checkpoint \
        --device-id $device_id \
        --dir exp/chain/inference/$x \
        --feat-dim $feat_dim \
        --feats-scp data/fbank_pitch/$x/feats.scp \
        --hidden-dim $hidden_dim \
        --is-training 0 \
        --kernel-size-list "$kernel_size_list" \
        --log-level $log_level \
        --model-left-context $model_left_context \
        --model-right-context $model_right_context \
        --output-dim $output_dim \
        --stride-list "$stride_list" || exit 1
    fi
  done
fi

if [[ $stage -le 26 ]]; then
  echo "decoding"
  for x in test dev; do
    if [[ ! -f exp/chain/inference/$x/confidence.scp ]]; then
      echo "exp/chain/inference/$x/confidence.scp does not exist!"
      echo "Please run inference.py first"
      exit 1
    fi
    echo "decoding $x"

    ./local/decode.sh \
      --nj $nj \
      exp/chain/graph \
      exp/chain/0.trans_mdl \
      exp/chain/inference/$x/confidence.scp \
      exp/chain/decode_res/$x
  done
fi

if [[ $stage -le 27 ]]; then
  echo "scoring"

  for x in test dev; do
    ./local/score.sh --cmd $decode_cmd \
      data/fbank_pitch/$x \
      exp/chain/graph \
      exp/chain/decode_res/$x || exit 1
  done

  for x in test dev; do
    head exp/chain/decode_res/$x/scoring_kaldi/best_*
  done
fi
