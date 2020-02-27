#!/usr/bin/env bash

# Copyright 2017-2018  Johns Hopkins University (author: Daniel Povey)
#           2017-2018  Yiming Wang

# tdnn-lstm recipe
# [for swahili]
# cat exp/chain/tdnn_lstm1a_sp/decode_dev/scoring_kaldi/best_wer
# %WER 39.12 [ 24312 / 62144, 3118 ins, 5952 del, 15242 sub ] exp/chain/tdnn_lstm1a_sp/decode_dev/wer_9_0.5

# steps/info/chain_dir_info.pl exp/chain/tdnn_lstm1a_sp
# exp/chain/tdnn_lstm1a_sp: num-iters=70 nj=2..12 num-params=10.9M dim=40+100->1792 combine=-0.176->-0.174 (over 6) xent:train/valid[45,69,final]=(-1.71,-1.52,-1.50/-1.81,-1.69,-1.67) logprob:train/valid[45,69,final]=(-0.185,-0.160,-0.159/-0.213,-0.208,-0.205)

# Set -e here so that we catch if any executable fails immediately
set -euo pipefail

# First the options that are passed through to run_ivector_common.sh
# (some of which are also used in this script directly).
stage=0
decode_nj=30
train_set=train
test_sets=dev
gmm=tri3
nnet3_affix=

# The rest are configs specific to this script.  Most of the parameters
# are just hardcoded at this level, in the commands below.
tlstm_affix=1a   # affix for the TDNN-LSTM directory name
tree_affix=
train_stage=-10
get_egs_stage=-10
decode_iter=

# training options
# training chunk-options
chunk_width=140,100,160
chunk_left_context=40
chunk_right_context=0
label_delay=5
common_egs_dir=
xent_regularize=0.1

# training options
srand=0
remove_egs=true
reporting_email=


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

# The iVector-extraction and feature-dumping parts are the same as the standard
# nnet3 setup, and you can skip them by setting "--stage 11" if you have already
# run those things.
local/nnet3/run_ivector_common.sh --stage $stage \
                                  --train-set $train_set \
                                  --gmm $gmm \
                                  --nnet3-affix "$nnet3_affix" || exit 1;

# Problem: We have removed the "train_" prefix of our training set in
# the alignment directory names! Bad!
gmm_dir=exp/$gmm
ali_dir=exp/${gmm}_ali_${train_set}_sp
tree_dir=exp/chain${nnet3_affix}/tree_sp${tree_affix:+_$tree_affix}
lang_combined=data/lang_combined_chain
lat_dir=exp/chain${nnet3_affix}/${gmm}_${train_set}_sp_lats
dir=exp/chain${nnet3_affix}/tdnn_lstm${tlstm_affix}_sp
train_data_dir=data/${train_set}_sp_hires
lores_train_data_dir=data/${train_set}_sp
train_ivector_dir=exp/nnet3${nnet3_affix}/ivectors_${train_set}_sp_hires

for f in $gmm_dir/final.mdl $train_data_dir/feats.scp $train_ivector_dir/ivector_online.scp \
    $lores_train_data_dir/feats.scp $ali_dir/ali.1.gz; do
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
  tdnn_opts="l2-regularize=0.02"
  lstm_opts="l2-regularize=0.005"
  output_opts="l2-regularize=0.004"

  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig
  input dim=100 name=ivector
  input dim=40 name=input

  # please note that it is important to have input layer with the name=input
  # as the layer immediately preceding the fixed-affine-layer to enable
  # the use of short notation for the descriptor
  fixed-affine-layer name=lda input=Append(-1,0,1,ReplaceIndex(ivector, t, 0)) affine-transform-file=$dir/configs/lda.mat

  # the first splicing is moved before the lda layer, so no splicing here
  relu-batchnorm-layer name=tdnn1 $tdnn_opts dim=512
  relu-batchnorm-layer name=tdnn2 $tdnn_opts input=Append(-1,0,1) dim=512
  relu-batchnorm-layer name=tdnn3 $tdnn_opts input=Append(-1,0,1) dim=512

  # check steps/libs/nnet3/xconfig/lstm.py for the other options and defaults
  fast-lstmp-layer name=lstm1 cell-dim=512 recurrent-projection-dim=128 non-recurrent-projection-dim=128 delay=-3 $lstm_opts
  relu-batchnorm-layer name=tdnn4 $tdnn_opts input=Append(-3,0,3) dim=512
  relu-batchnorm-layer name=tdnn5 $tdnn_opts input=Append(-3,0,3) dim=512
  relu-batchnorm-layer name=tdnn6 $tdnn_opts input=Append(-3,0,3) dim=512
  fast-lstmp-layer name=lstm2 cell-dim=512 recurrent-projection-dim=128 non-recurrent-projection-dim=128 delay=-3 $lstm_opts
  relu-batchnorm-layer name=tdnn7 $tdnn_opts input=Append(-3,0,3) dim=512
  relu-batchnorm-layer name=tdnn8 $tdnn_opts input=Append(-3,0,3) dim=512
  relu-batchnorm-layer name=tdnn9 $tdnn_opts input=Append(-3,0,3) dim=512
  fast-lstmp-layer name=lstm3 cell-dim=512 recurrent-projection-dim=128 non-recurrent-projection-dim=128 delay=-3 $lstm_opts

  ## adding the layers for chain branch
  output-layer name=output input=lstm3 output-delay=$label_delay include-log-softmax=false dim=$num_targets max-change=1.5 $output_opts

  # adding the layers for xent branch
  # This block prints the configs for a separate output that will be
  # trained with a cross-entropy objective in the 'chain' models... this
  # has the effect of regularizing the hidden parts of the model.  we use
  # 0.5 / args.xent_regularize as the learning rate factor- the factor of
  # 0.5 / args.xent_regularize is suitable as it means the xent
  # final-layer learns at a rate independent of the regularization
  # constant; and the 0.5 was tuned so as to make the relative progress
  # similar in the xent and regular final layers.
  output-layer name=output-xent input=lstm3 output-delay=$label_delay dim=$num_targets learning-rate-factor=$learning_rate_factor max-change=1.5 $output_opts
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
    --chain.l2-regularize=0.00005 \
    --chain.apply-deriv-weights=false \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --trainer.srand=$srand \
    --trainer.max-param-change=2.0 \
    --trainer.num-epochs=5 \
    --trainer.frames-per-iter=1500000 \
    --trainer.optimization.num-jobs-initial=2 \
    --trainer.optimization.num-jobs-final=12 \
    --trainer.optimization.initial-effective-lrate=0.001 \
    --trainer.optimization.final-effective-lrate=0.0001 \
    --trainer.optimization.shrink-value=1.0 \
    --trainer.num-chunk-per-minibatch=128,64 \
    --trainer.optimization.momentum=0.0 \
    --trainer.deriv-truncate-margin=8 \
    --egs.chunk-width=$chunk_width \
    --egs.chunk-left-context=$chunk_left_context \
    --egs.chunk-right-context=$chunk_right_context \
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
    --self-loop-scale 1.0 data/lang_test \
    $tree_dir $tree_dir/graph || exit 1;

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
          --extra-left-context $chunk_left_context \
          --extra-right-context $chunk_right_context \
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
