#!/usr/bin/env bash

# Copyright    2017 Hossein Hadian
#              2017 Chun Chieh Chang
#              2017 Ashish Arora

# steps/info/chain_dir_info.pl exp/chain/cnn1a/
# exp/chain/cnn1a/: num-iters=153 nj=3..10 num-params=3.6M dim=40->268 combine=-0.034->-0.034 xent:train/valid[101,152,final]=(-0.097,-0.186,-0.092/-0.101,-0.212,-0.098) logprob:train/valid[101,152,final]=(-0.035,-0.067,-0.035/-0.036,-0.082,-0.035)

# cat exp/chain/cnn1a/decode_test/scoring_kaldi/best_*
# %WER 0.19 [ 366 / 188135, 110 ins, 123 del, 133 sub ] exp/chain/cnn1a/decode_test/cer_7_0.5
# %WER 1.00 [ 357 / 35571, 104 ins, 26 del, 227 sub ] exp/chain/cnn1a/decode_test/wer_5_1.0


set -e -o pipefail

stage=0
nj=30

# affix for exp dirs, e.g. it was _cleaned in tedlium.
nnet3_affix=

affix=1a
common_egs_dir=
reporting_email=

# chain options
train_stage=-10
xent_regularize=0.1
frame_subsampling_factor=5

# training chunk-options
chunk_width=340,300,200,100

# we don't need extra left/right context for TDNN systems.
chunk_left_context=0
chunk_right_context=0

# training options
srand=0
remove_egs=false

gmm_dir=exp/tri2
ali_dir=exp/tri2_ali
lat_dir=exp/chain${nnet3_affix}/tri2_train_lats
dir=exp/chain${nnet3_affix}/cnn${affix}
train_data_dir=data/train
lores_train_data_dir=$train_data_dir  # for the start, use the same data for gmm and chain
lang_test=data/lang_unk
tree_dir=exp/chain${nnet3_affix}/tree${affix}

# the 'lang' directory is created by this script.
# If you create such a directory with a non-standard topology
# you should probably name it differently.
lang=data/lang_chain

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

for f in $train_data_dir/feats.scp \
    $lores_train_data_dir/feats.scp $gmm_dir/final.mdl \
    $ali_dir/ali.1.gz $gmm_dir/final.mdl; do
  [ ! -f $f ] && echo "$0: expected file $f to exist" && exit 1
done


if [ $stage -le 1 ]; then
  echo "$0: creating lang directory $lang with chain-type topology"
  # Create a version of the lang/ directory that has one state per phone in the
  # topo file. [note, it really has two states.. the first one is only repeated
  # once, the second one has zero or more repeats.]
  if [ -d $lang ]; then
    if [ $lang/L.fst -nt data/lang/L.fst ]; then
      echo "$0: $lang already exists, not overwriting it; continuing"
    else
      echo "$0: $lang already exists and seems to be older than data/lang..."
      echo " ... not sure what to do.  Exiting."
      exit 1;
    fi
  else
    cp -r data/lang $lang
    silphonelist=$(cat $lang/phones/silence.csl) || exit 1;
    nonsilphonelist=$(cat $lang/phones/nonsilence.csl) || exit 1;
    # Use our special topology... note that later on may have to tune this
    # topology.
    steps/nnet3/chain/gen_topo.py $nonsilphonelist $silphonelist >$lang/topo
  fi
fi

if [ $stage -le 2 ]; then
  # Get the alignments as lattices (gives the chain training more freedom).
  # use the same num-jobs as the alignments
  steps/align_fmllr_lats.sh --nj $nj --cmd "$cmd" ${lores_train_data_dir} \
    data/lang $gmm_dir $lat_dir
  rm $lat_dir/fsts.*.gz # save space
fi

if [ $stage -le 3 ]; then
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
    --cmd "$cmd" 300 ${lores_train_data_dir} \
    $lang $ali_dir $tree_dir
fi


if [ $stage -le 4 ]; then
  mkdir -p $dir
  echo "$0: creating neural net configs using the xconfig parser";

  num_targets=$(tree-info $tree_dir/tree | grep num-pdfs | awk '{print $2}')
  learning_rate_factor=$(echo "print (0.5/$xent_regularize)" | python)
  common1="required-time-offsets=0 height-offsets=-2,-1,0,1,2 num-filters-out=12"

  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig
  input dim=40 name=input

  conv-relu-batchnorm-layer name=cnn1 height-in=40 height-out=40 time-offsets=-2,-1,0,1,2 $common1
  conv-relu-batchnorm-layer name=cnn2 height-in=40 height-out=40 time-offsets=-2,-1,0,1,2 $common1
  relu-batchnorm-layer name=tdnn1 input=Append(-2,-1,0,1,2) dim=450
  relu-batchnorm-layer name=tdnn2 input=Append(-5,0,5) dim=450
  relu-batchnorm-layer name=tdnn3 input=Append(-5,0,5) dim=450
  relu-batchnorm-layer name=tdnn4 input=Append(-5,0,5) dim=450

  ## adding the layers for chain branch
  relu-batchnorm-layer name=prefinal-chain dim=450 target-rms=0.5
  output-layer name=output include-log-softmax=false dim=$num_targets max-change=1.5

  # adding the layers for xent branch
  # This block prints the configs for a separate output that will be
  # trained with a cross-entropy objective in the 'chain' models... this
  # has the effect of regularizing the hidden parts of the model.  we use
  # 0.5 / args.xent_regularize as the learning rate factor- the factor of
  # 0.5 / args.xent_regularize is suitable as it means the xent
  # final-layer learns at a rate independent of the regularization
  # constant; and the 0.5 was tuned so as to make the relative progress
  # similar in the xent and regular final layers.
  relu-batchnorm-layer name=prefinal-xent input=tdnn4 dim=450 target-rms=0.5
  output-layer name=output-xent dim=$num_targets learning-rate-factor=$learning_rate_factor max-change=1.5
EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
fi


if [ $stage -le 5 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{3,4,5,6}/$USER/kaldi-data/egs/iam-$(date +'%m_%d_%H_%M')/s5/$dir/egs/storage $dir/egs/storage
  fi

  steps/nnet3/chain/train.py --stage=$train_stage \
    --cmd="$cmd" \
    --feat.cmvn-opts="--norm-means=false --norm-vars=false" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient=0.1 \
    --chain.l2-regularize=0.00005 \
    --chain.apply-deriv-weights=false \
    --chain.lm-opts="--num-extra-lm-states=500" \
    --chain.frame-subsampling-factor=$frame_subsampling_factor \
    --chain.alignment-subsampling-factor=$frame_subsampling_factor \
    --trainer.srand=$srand \
    --trainer.max-param-change=2.0 \
    --trainer.num-epochs=4 \
    --trainer.frames-per-iter=1500000 \
    --trainer.optimization.num-jobs-initial=3 \
    --trainer.optimization.num-jobs-final=10 \
    --trainer.optimization.initial-effective-lrate=0.001 \
    --trainer.optimization.final-effective-lrate=0.0001 \
    --trainer.optimization.shrink-value=1.0 \
    --trainer.num-chunk-per-minibatch=128,64 \
    --trainer.optimization.momentum=0.0 \
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

if [ $stage -le 6 ]; then
  # The reason we are using data/lang here, instead of $lang, is just to
  # emphasize that it's not actually important to give mkgraph.sh the
  # lang directory with the matched topology (since it gets the
  # topology file from the model).  So you could give it a different
  # lang directory, one that contained a wordlist and LM of your choice,
  # as long as phones.txt was compatible.

  utils/mkgraph.sh \
    --self-loop-scale 1.0 $lang_test \
    $dir $dir/graph || exit 1;
fi

if [ $stage -le 7 ]; then
  frames_per_chunk=$(echo $chunk_width | cut -d, -f1)
  steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
    --extra-left-context $chunk_left_context \
    --extra-right-context $chunk_right_context \
    --extra-left-context-initial 0 \
    --extra-right-context-final 0 \
    --frames-per-chunk $frames_per_chunk \
    --nj $nj --cmd "$cmd" \
    $dir/graph data/test $dir/decode_test || exit 1;
fi
