#!/usr/bin/env bash

# chainali_1a is as 1a except it uses chain alignments (using 1a system) instead of gmm alignments

# ./local/chain/compare_wer.sh exp/chain/cnn_chainali_1a/ exp/chain/cnn_1a/

# steps/info/chain_dir_info.pl exp/chain/cnn_chainali_1a/
# exp/chain/cnn_chainali_1a/: num-iters=21 nj=2..4 num-params=4.4M dim=40->364 combine=-0.002->0.000 xent:train/valid[13,20,final]=(-0.929,-0.711,-0.645/-1.16,-1.04,-0.992) logprob:train/valid[13,20,final]=(-0.029,-0.016,-0.013/-0.051,-0.047,-0.045)

# cat exp/chain/cnn_chainali_1a/decode_test/scoring_kaldi/best_*

set -e -o pipefail

stage=0
nj=30
train_set=train
gmm=tri3        # this is the source gmm-dir that we'll use for alignments; it
                # should have alignments for the specified training data.
nnet3_affix=    # affix for exp dirs, e.g. it was _cleaned in tedlium.
affix=_1a  #affix for TDNN+LSTM directory e.g. "1a" or "1b", in case we change the configuration.
ali=tri3_ali
chain_model_dir=exp/chain${nnet3_affix}/cnn${affix}
common_egs_dir=
reporting_email=

# chain options
train_stage=-10
xent_regularize=0.1
# training chunk-options
chunk_width=340,300,200,100
num_leaves=500
# we don't need extra left/right context for TDNN systems.
tdnn_dim=450
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

gmm_dir=exp/${gmm}
ali_dir=exp/${ali}
lat_dir=exp/chain${nnet3_affix}/${gmm}_${train_set}_lats_chain
gmm_lat_dir=exp/chain${nnet3_affix}/${gmm}_${train_set}_lats
dir=exp/chain${nnet3_affix}/cnn_chainali${affix}
train_data_dir=data/${train_set}
tree_dir=exp/chain${nnet3_affix}/tree_chain

# the 'lang' directory is created by this script.
# If you create such a directory with a non-standard topology
# you should probably name it differently.
lang=data/lang_chain
for f in $train_data_dir/feats.scp \
    $train_data_dir/feats.scp $gmm_dir/final.mdl \
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
  steps/nnet3/align_lats.sh --nj $nj --cmd "$cmd" \
                            --scale-opts '--transition-scale=1.0 --self-loop-scale=1.0' \
                            ${train_data_dir} data/lang $chain_model_dir $lat_dir
  cp $gmm_lat_dir/splice_opts $lat_dir/splice_opts
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
    --frame-subsampling-factor 4 \
    --context-opts "--context-width=2 --central-position=1" \
    --cmd "$cmd" $num_leaves ${train_data_dir} \
    $lang $ali_dir $tree_dir
fi


if [ $stage -le 4 ]; then
  mkdir -p $dir
  echo "$0: creating neural net configs using the xconfig parser";

  num_targets=$(tree-info $tree_dir/tree | grep num-pdfs | awk '{print $2}')
  learning_rate_factor=$(echo "print (0.5/$xent_regularize)" | python)
  common1="height-offsets=-2,-1,0,1,2 num-filters-out=36"
  common2="height-offsets=-2,-1,0,1,2 num-filters-out=70"
  common3="height-offsets=-1,0,1 num-filters-out=70"
  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig
  input dim=40 name=input
  conv-relu-batchnorm-layer name=cnn1 height-in=40 height-out=40 time-offsets=-3,-2,-1,0,1,2,3 $common1
  conv-relu-batchnorm-layer name=cnn2 height-in=40 height-out=20 time-offsets=-2,-1,0,1,2 $common1 height-subsample-out=2
  conv-relu-batchnorm-layer name=cnn3 height-in=20 height-out=20 time-offsets=-4,-2,0,2,4 $common2
  conv-relu-batchnorm-layer name=cnn4 height-in=20 height-out=10 time-offsets=-4,-2,0,2,4 $common2 height-subsample-out=2
  relu-batchnorm-layer name=tdnn1 input=Append(-4,-2,0,2,4) dim=$tdnn_dim
  relu-batchnorm-layer name=tdnn2 input=Append(-4,0,4) dim=$tdnn_dim
  relu-batchnorm-layer name=tdnn3 input=Append(-4,0,4) dim=$tdnn_dim
  relu-batchnorm-layer name=tdnn4 input=Append(-4,0,4) dim=$tdnn_dim

  ## adding the layers for chain branch
  relu-batchnorm-layer name=prefinal-chain dim=$tdnn_dim target-rms=0.5
  output-layer name=output include-log-softmax=false dim=$num_targets max-change=1.5

  # adding the layers for xent branch
  # This block prints the configs for a separate output that will be
  # trained with a cross-entropy objective in the 'chain' mod?els... this
  # has the effect of regularizing the hidden parts of the model.  we use
  # 0.5 / args.xent_regularize as the learning rate factor- the factor of
  # 0.5 / args.xent_regularize is suitable as it means the xent
  # final-layer learns at a rate independent of the regularization
  # constant; and the 0.5 was tuned so as to make the relative progress
  # similar in the xent and regular final layers.
  relu-batchnorm-layer name=prefinal-xent input=tdnn4 dim=$tdnn_dim target-rms=0.5
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
    --chain.frame-subsampling-factor=4 \
    --chain.alignment-subsampling-factor=1 \
    --trainer.srand=0 \
    --trainer.max-param-change=2.0 \
    --trainer.num-epochs=2 \
    --trainer.frames-per-iter=2000000 \
    --trainer.optimization.num-jobs-initial=8 \
    --trainer.optimization.num-jobs-final=16 \
    --trainer.optimization.initial-effective-lrate=0.001 \
    --trainer.optimization.final-effective-lrate=0.0001 \
    --trainer.optimization.shrink-value=1.0 \
    --trainer.num-chunk-per-minibatch=64,32 \
    --trainer.optimization.momentum=0.0 \
    --egs.chunk-width=$chunk_width \
    --egs.dir="$common_egs_dir" \
    --egs.opts="--frames-overlap-per-eg 0" \
    --cleanup.remove-egs=false \
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
    --self-loop-scale 1.0 data/lang_test \
    $dir $dir/graph || exit 1;
fi

if [ $stage -le 7 ]; then
  frames_per_chunk=$(echo $chunk_width | cut -d, -f1)
  steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
    --frames-per-chunk $frames_per_chunk \
    --nj $nj --cmd "$cmd" \
    $dir/graph data/test $dir/decode_test || exit 1;
fi
