#!/usr/bin/env bash

# Copyright 2017-2018  Johns Hopkins University (author: Daniel Povey)
#           2017-2018  Yiming Wang

# 1a is trying an architecture with factored parameter matrices with dropout.

# cat exp/chain/tdnn1b_sp/decode_dev/scoring_kaldi/best_wer
# [for swahili]
# %WER 36.84 [ 22893 / 62144, 2988 ins, 5712 del, 14193 sub ] exp/chain/tdnn1b_sp/decode_dev/wer_10_0.0
# [for tagalog]
# %WER 46.37 [ 29852 / 64382, 4163 ins, 7652 del, 18037 sub ] exp/chain/tdnn1b_sp/decode_dev/wer_10_0.0
# [for somali]
# %WER 57.44 [ 46889 / 81637, 5016 ins, 12015 del, 29858 sub ] exp/chain/tdnn1b_sp/decode_dev/wer_10_0.0

# steps/info/chain_dir_info.pl exp/chain/tdnn1b_sp

# [for swahili]
# exp/chain/tdnn1b_sp/: num-iters=99 nj=2..12 num-params=17.2M dim=40+100->1816
# combine=-0.127->-0.127 (over 2) xent:train/valid[65,98,final]=(-1.74,-1.44,-1.43/-1.80,-1.62,-1.61)
# logprob:train/valid[65,98,final]=(-0.175,-0.136,-0.135/-0.194,-0.182,-0.180)

# [for tagalog]
# exp/chain/tdnn1b_sp/: num-iters=96 nj=2..12 num-params=17.2M dim=40+100->1928 combine=-0.124->-0.123
# (over 2) xent:train/valid[63,95,final]=(-1.69,-1.43,-1.42/-1.75,-1.62,-1.60) 
# logprob:train/valid[63,95,final]=(-0.168,-0.128,-0.127/-0.193,-0.187,-0.187)

# [for somali]
# exp/chain/tdnn1b_sp/: num-iters=84 nj=2..12 num-params=17.9M dim=40+100->3240 combine=-0.162->-0.160 
# (over 2) xent:train/valid[55,83,final]=(-2.31,-2.02,-2.00/-2.27,-2.13,-2.10)
# logprob:train/valid[55,83,final]=(-0.218,-0.157,-0.154/-0.268,-0.263,-0.263)

set -e -o pipefail

# First the options that are passed through to run_ivector_common.sh
# (some of which are also used in this script directly).
stage=0
nj=30
train_set=train
test_sets="dev"
gmm=tri3        # this is the source gmm-dir that we'll use for alignments; it
                 # should have alignments for the specified training data.
nnet3_affix=       # affix for exp dirs, e.g. it was _cleaned in tedlium.

# Options which are not passed through to run_ivector_common.sh
affix=1b   #affix for TDNN directory e.g. "1a" or "1b", in case we change the configuration.
tree_affix=
common_egs_dir=
reporting_email=

# LSTM/chain options
train_stage=-10
get_egs_stage=-10
xent_regularize=0.1

# training chunk-options
chunk_width=140,100,160
# we don't need extra left/right context for TDNN systems.
chunk_left_context=0
chunk_right_context=0
dropout_schedule='0,0@0.20,0.3@0.50,0'
num_epochs=7

# training options
srand=0
remove_egs=true


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

local/nnet3/run_ivector_common.sh \
  --stage $stage --nj $nj \
  --train-set $train_set --gmm $gmm \
  --nnet3-affix "$nnet3_affix" || exit 1;


gmm_dir=exp/${gmm}
ali_dir=exp/${gmm}_ali_${train_set}_sp
tree_dir=exp/chain${nnet3_affix}/tree_sp${tree_affix:+_$tree_affix}
lang_combined=data/lang_combined_chain
lat_dir=exp/chain${nnet3_affix}/${gmm}_${train_set}_sp_lats
dir=exp/chain${nnet3_affix}/tdnn${affix}_sp
train_data_dir=data/${train_set}_sp_hires
train_ivector_dir=exp/nnet3${nnet3_affix}/ivectors_${train_set}_sp_hires
lores_train_data_dir=data/${train_set}_sp

for f in $train_data_dir/feats.scp $train_ivector_dir/ivector_online.scp \
    $lores_train_data_dir/feats.scp $gmm_dir/final.mdl \
    $ali_dir/ali.1.gz $gmm_dir/final.mdl; do
  [ ! -f $f ] && echo "$0: expected file $f to exist" && exit 1
done

if [ $stage -le 7 ]; then
  echo "$0: creating lang directory $lang_combined with chain-type topology"
  # Create a version of the lang/ directory that has one state per phone in the
  # topo file. [note, it really has two states.. the first one is only repeated
  # once, the second one has zero or more repeats.]
  if [ -d $lang_combined ]; then
    if [ $lang_combined/L.fst -nt data/lang_combined_test/L.fst ]; then
      echo "$0: $lang_combined already exists, not overwriting it; continuing"
    else
      echo "$0: $lang_combined already exists and seems to be older than data/lang_combined_test ..."
      echo " ... not sure what to do.  Exiting."
      exit 1;
    fi
  else
    cp -r data/lang_combined_test $lang_combined
    silphonelist=$(cat $lang_combined/phones/silence.csl) || exit 1;
    nonsilphonelist=$(cat $lang_combined/phones/nonsilence.csl) || exit 1;
    # Use our special topology... note that later on may have to tune this
    # topology.
    steps/nnet3/chain/gen_topo.py $nonsilphonelist $silphonelist >$lang_combined/topo
  fi
fi

if [ $stage -le 8 ]; then
  # Get the alignments as lattices (gives the chain training more freedom).
  # use the same num-jobs as the alignments
  steps/align_fmllr_lats.sh --nj 75 --cmd "$train_cmd" ${lores_train_data_dir} \
    data/lang_combined $gmm_dir $lat_dir
  rm $lat_dir/fsts.*.gz # save space
fi

if [ $stage -le 9 ]; then
  # Build a tree using our new topology.  We know we have alignments for the
  # speed-perturbed data (local/nnet3/run_ivector_common.sh made them), so use
  # those.  The num-leaves is always somewhat less than the num-leaves from
  # the GMM baseline.
  if [ -f $tree_dir/final.mdl ]; then
    echo "$0: $tree_dir/final.mdl already exists, refusing to overwrite it."
    exit 1;
  fi
  steps/nnet3/chain/build_tree.sh \
    --frame-subsampling-factor 3 \
    --context-opts "--context-width=2 --central-position=1" \
    --cmd "$train_cmd" 6000 ${lores_train_data_dir} \
    $lang_combined $ali_dir $tree_dir
fi


if [ $stage -le 10 ]; then
  mkdir -p $dir
  echo "$0: creating neural net configs using the xconfig parser";

  num_targets=$(tree-info $tree_dir/tree |grep num-pdfs|awk '{print $2}')
  learning_rate_factor=$(echo "print 0.5/$xent_regularize" | python)
  opts="l2-regularize=0.004 dropout-proportion=0.0 dropout-per-dim=true dropout-per-dim-continuous=true"
  linear_opts="orthonormal-constraint=-1.0 l2-regularize=0.004"
  output_opts="l2-regularize=0.002"

  mkdir -p $dir/configs

  cat <<EOF > $dir/configs/network.xconfig
  input dim=100 name=ivector
  input dim=40 name=input

  # please note that it is important to have input layer with the name=input
  # as the layer immediately preceding the fixed-affine-layer to enable
  # the use of short notation for the descriptor
  fixed-affine-layer name=lda input=Append(-1,0,1,ReplaceIndex(ivector, t, 0)) affine-transform-file=$dir/configs/lda.mat

  # the first splicing is moved before the lda layer, so no splicing here
  relu-batchnorm-dropout-layer name=tdnn1 $opts dim=1024
  linear-component name=tdnn2l0 dim=256 $linear_opts input=Append(-1,0)
  linear-component name=tdnn2l dim=256 $linear_opts input=Append(-1,0)
  relu-batchnorm-dropout-layer name=tdnn2 $opts input=Append(0,1) dim=1024
  linear-component name=tdnn3l dim=256 $linear_opts input=Append(-1,0)
  relu-batchnorm-dropout-layer name=tdnn3 $opts dim=1024 input=Append(0,1)
  linear-component name=tdnn4l0 dim=256 $linear_opts input=Append(-1,0)
  linear-component name=tdnn4l dim=256 $linear_opts input=Append(0,1)
  relu-batchnorm-dropout-layer name=tdnn4 $opts input=Append(0,1) dim=1024
  linear-component name=tdnn5l dim=256 $linear_opts
  relu-batchnorm-dropout-layer name=tdnn5 $opts dim=1024 input=Append(0, tdnn3l)
  linear-component name=tdnn6l0 dim=256 $linear_opts input=Append(-3,0)
  linear-component name=tdnn6l dim=256 $linear_opts input=Append(-3,0)
  relu-batchnorm-dropout-layer name=tdnn6 $opts input=Append(0,3) dim=1280
  linear-component name=tdnn7l0 dim=256 $linear_opts input=Append(-3,0)
  linear-component name=tdnn7l dim=256 $linear_opts input=Append(0,3)
  relu-batchnorm-dropout-layer name=tdnn7 $opts input=Append(0,3,tdnn6l,tdnn4l,tdnn2l) dim=1024
  linear-component name=tdnn8l0 dim=256 $linear_opts input=Append(-3,0)
  linear-component name=tdnn8l dim=256 $linear_opts input=Append(0,3)
  relu-batchnorm-dropout-layer name=tdnn8 $opts input=Append(0,3) dim=1280
  linear-component name=tdnn9l0 dim=256 $linear_opts input=Append(-3,0)
  linear-component name=tdnn9l dim=256 $linear_opts input=Append(-3,0)
  relu-batchnorm-dropout-layer name=tdnn9 $opts input=Append(0,3,tdnn8l,tdnn6l,tdnn5l) dim=1024
  linear-component name=tdnn10l0 dim=256 $linear_opts input=Append(-3,0)
  linear-component name=tdnn10l dim=256 $linear_opts input=Append(0,3)
  relu-batchnorm-dropout-layer name=tdnn10 $opts input=Append(0,3) dim=1280
  linear-component name=tdnn11l0 dim=256 $linear_opts input=Append(-3,0)
  linear-component name=tdnn11l dim=256 $linear_opts input=Append(-3,0)
  relu-batchnorm-dropout-layer name=tdnn11 $opts input=Append(0,3,tdnn10l,tdnn9l,tdnn7l) dim=1024
  linear-component name=prefinal-l dim=256 $linear_opts

  relu-batchnorm-layer name=prefinal-chain input=prefinal-l $opts dim=1280
  linear-component name=prefinal-chain-l dim=256 $linear_opts
  batchnorm-component name=prefinal-chain-batchnorm
  output-layer name=output include-log-softmax=false dim=$num_targets $output_opts

  relu-batchnorm-layer name=prefinal-xent input=prefinal-l $opts dim=1280
  linear-component name=prefinal-xent-l dim=256 $linear_opts
  batchnorm-component name=prefinal-xent-batchnorm
  output-layer name=output-xent dim=$num_targets learning-rate-factor=$learning_rate_factor $output_opts
  
EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
fi


if [ $stage -le 11 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{3,4,5,6}/$USER/kaldi-data/egs/material-$(date +'%m_%d_%H_%M')/s5/$dir/egs/storage $dir/egs/storage
  fi

  steps/nnet3/chain/train.py --stage=$train_stage \
    --cmd="$decode_cmd" \
    --feat.online-ivector-dir=$train_ivector_dir \
    --feat.cmvn-opts="--norm-means=false --norm-vars=false" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient=0.1 \
    --chain.l2-regularize=0.0 \
    --chain.apply-deriv-weights=false \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --trainer.dropout-schedule $dropout_schedule \
    --trainer.add-option="--optimization.memory-compression-level=2" \
    --trainer.srand=$srand \
    --trainer.max-param-change=2.0 \
    --trainer.num-epochs=$num_epochs \
    --trainer.frames-per-iter=1500000 \
    --trainer.optimization.num-jobs-initial=2 \
    --trainer.optimization.num-jobs-final=12 \
    --trainer.optimization.initial-effective-lrate=0.001 \
    --trainer.optimization.final-effective-lrate=0.0001 \
    --trainer.num-chunk-per-minibatch=128,64 \
    --trainer.optimization.momentum=0.0 \
    --egs.chunk-width=$chunk_width \
    --egs.chunk-left-context=0 \
    --egs.chunk-right-context=0 \
    --egs.chunk-left-context-initial=0 \
    --egs.chunk-right-context-final=0 \
    --egs.dir="$common_egs_dir" \
    --egs.opts="--frames-overlap-per-eg 0" \
    --cleanup.remove-egs=$remove_egs \
    --use-gpu=true \
    --reporting.email="$reporting_email" \
    --feat-dir=$train_data_dir \
    --tree-dir=$tree_dir \
    --lat-dir=$lat_dir \
    --dir=$dir  || exit 1;
fi

if [ $stage -le 12 ]; then
  # Note: it's not important to give mkgraph.sh the lang directory with the
  # matched topology (since it gets the topology file from the model).
  utils/mkgraph.sh \
    --self-loop-scale 1.0 data/lang_combined_test \
    $tree_dir ${tree_dir}/graph_combined || exit 1;
fi

if [ $stage -le 13 ]; then
  frames_per_chunk=$(echo $chunk_width | cut -d, -f1)
  rm $dir/.error 2>/dev/null || true

  for data in $test_sets; do
    (
      nspk=$(wc -l <data/${data}_hires/spk2utt)
      steps/nnet3/decode.sh \
          --acwt 1.0 --post-decode-acwt 10.0 \
          --extra-left-context 0 --extra-right-context 0 \
          --extra-left-context-initial 0 \
          --extra-right-context-final 0 \
          --frames-per-chunk $frames_per_chunk \
          --nj $nspk --cmd "$decode_cmd"  --num-threads 4 \
          --online-ivector-dir exp/nnet3${nnet3_affix}/ivectors_${data}_hires \
          $tree_dir/graph_combined data/${data}_hires ${dir}/decode_${data} || exit 1
    ) || touch $dir/.error &
  done
  wait
  [ -f $dir/.error ] && echo "$0: there was a problem while decoding" && exit 1
fi

exit 0;
