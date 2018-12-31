#!/bin/bash

# e2eali_1a is a 6 cnn layer 3 tdnn layer model with dropout, l2-regularization, batch-normalization

# local/chain/compare_wer.sh exp/chain/cnn_e2eali_1a
# System                      cnn_e2eali_1a
# WER                              7.75
# CER                              2.68
# Final train prob              -0.0779
# Final valid prob              -0.0860
# Final train prob (xent)       -0.7744
# Final valid prob (xent)       -0.8111
# Parameters                      4.96M

# steps/info/chain_dir_info.pl exp/chain/cnn_e2eali_1a
# exp/chain/cnn_e2eali_1a: num-iters=36 nj=3..8 num-params=5.0M dim=40->944 combine=-0.076->-0.076 (over 1) xent:train/valid[23,35,final]=(-1.48,-0.871,-0.774/-1.46,-0.888,-0.811) logprob:train/valid[23,35,final]=(-0.208,-0.102,-0.078/-0.189,-0.104,-0.086)

# line level scoring result
# WER 7.75 [ 437 / 5639, 62 ins, 55 del, 320 sub ] exp/chain/cnn_e2eali_1d/decode_test/wer_7_1.0
# paragraph scoring result
# WER 6.69 [ 377 / 5639, 44 ins, 37 del, 296 sub ] exp/chain/cnn_e2eali_1a/decode_test/para/wer_7_1.0

set -e -o pipefail

stage=0

nj=50
train_set=train
decode_val=true
nnet3_affix=    # affix for exp dirs, e.g. it was _cleaned in tedlium.
affix=_1a  #affix for TDNN+LSTM directory e.g. "1a" or "1b", in case we change the configuration.
e2echain_model_dir=exp/chain/e2e_cnn_1a
tree_affix=_1a
bnf_chain_model_dir=exp/chain/e2e_cnn_1a
bnf_layer_name=tdnn6.affine
common_egs_dir=
reporting_email=

# chain options
train_stage=-10
xent_regularize=0.1
# training chunk-options
chunk_width=340,300,200,100
num_leaves=1000
# we don't need extra left/right context for TDNN systems.
tdnn_dim=550
# training options
srand=0
remove_egs=true
lang_decode=data/lang
if $decode_val; then maybe_val=val; else maybe_val= ; fi
dropout_schedule='0,0@0.20,0.2@0.50,0'
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

ali_dir=exp/chain/e2e_ali_train
lat_dir=exp/chain${nnet3_affix}/e2e_${train_set}_lats
dir=exp/chain${nnet3_affix}/cnn_e2eali${affix}
train_data_dir=data/${train_set}
tree_dir=exp/chain${nnet3_affix}/tree_e2e${tree_affix}

# the 'lang' directory is created by this script.
# If you create such a directory with a non-standard topology
# you should probably name it differently.
lang=data/lang_chain
for f in $train_data_dir/feats.scp; do
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
                            --acoustic-scale 1.0 --generate-ali-from-lats true \
                            --scale-opts '--transition-scale=1.0 --self-loop-scale=1.0' \
                            ${train_data_dir} data/lang $e2echain_model_dir $lat_dir
  echo "" >$lat_dir/splice_opts
fi

bnf_data_dir=$bnf_chain_model_dir/$(basename $train_data_dir)
if [ $stage -le 3 ]; then
  if [ -f $bnf_data_dir/feats.scp ]; then
    echo "$0: $bnf_data_dir/feats.scp exists. Refusing to dump features!"
    exit 1
  fi

  steps/nnet3/make_bottleneck_features.sh --cmd "$cmd" --use-gpu true \
    --compress false --nj $nj \
    $bnf_layer_name ${train_data_dir} ${bnf_data_dir} $bnf_chain_model_dir || exit 1
fi

if [ $stage -le 4 ]; then
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
    --alignment-subsampling-factor 1 \
    --context-opts "--context-width=2 --central-position=1" \
    --cmd "$cmd" $num_leaves ${bnf_data_dir} \
    $lang $lat_dir $tree_dir
fi


if [ $stage -le 5 ]; then
  mkdir -p $dir
  echo "$0: creating neural net configs using the xconfig parser";

  num_targets=$(tree-info $tree_dir/tree | grep num-pdfs | awk '{print $2}')
  learning_rate_factor=$(echo "print 0.5/$xent_regularize" | python)
  cnn_opts="l2-regularize=0.03 dropout-proportion=0.0"
  tdnn_opts="l2-regularize=0.03"
  output_opts="l2-regularize=0.04"
  common1="$cnn_opts required-time-offsets= height-offsets=-2,-1,0,1,2 num-filters-out=36"
  common2="$cnn_opts required-time-offsets= height-offsets=-2,-1,0,1,2 num-filters-out=70"
  common3="$cnn_opts required-time-offsets= height-offsets=-1,0,1 num-filters-out=70"
  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig
  input dim=40 name=input
  conv-relu-batchnorm-dropout-layer name=cnn1 height-in=40 height-out=40 time-offsets=-3,-2,-1,0,1,2,3 $common1
  conv-relu-batchnorm-dropout-layer name=cnn2 height-in=40 height-out=20 time-offsets=-2,-1,0,1,2 $common1 height-subsample-out=2
  conv-relu-batchnorm-dropout-layer name=cnn3 height-in=20 height-out=20 time-offsets=-4,-2,0,2,4 $common2
  conv-relu-batchnorm-dropout-layer name=cnn4 height-in=20 height-out=20 time-offsets=-4,-2,0,2,4 $common2
  conv-relu-batchnorm-dropout-layer name=cnn5 height-in=20 height-out=10 time-offsets=-4,-2,0,2,4 $common3 height-subsample-out=2
  conv-relu-batchnorm-dropout-layer name=cnn6 height-in=10 height-out=10 time-offsets=-4,0,4 $common3
  relu-batchnorm-dropout-layer name=tdnn1 input=Append(-4,0,4) dim=$tdnn_dim $tdnn_opts dropout-proportion=0.0
  relu-batchnorm-dropout-layer name=tdnn2 input=Append(-4,0,4) dim=$tdnn_dim $tdnn_opts dropout-proportion=0.0
  relu-batchnorm-dropout-layer name=tdnn3 input=Append(-4,0,4) dim=$tdnn_dim $tdnn_opts dropout-proportion=0.0
  ## adding the layers for chain branch
  relu-batchnorm-layer name=prefinal-chain dim=$tdnn_dim target-rms=0.5 $tdnn_opts
  output-layer name=output include-log-softmax=false dim=$num_targets max-change=1.5 $output_opts
  # adding the layers for xent branch
  # This block prints the configs for a separate output that will be
  # trained with a cross-entropy objective in the 'chain' models... this
  # has the effect of regularizing the hidden parts of the model.  we use
  # 0.5 / args.xent_regularize as the learning rate factor- the factor of
  # 0.5 / args.xent_regularize is suitable as it means the xent
  # final-layer learns at a rate independent of the regularization
  # constant; and the 0.5 was tuned so as to make the relative progress
  # similar in the xent and regular final layers.
  relu-batchnorm-layer name=prefinal-xent input=tdnn3 dim=$tdnn_dim target-rms=0.5 $tdnn_opts
  output-layer name=output-xent dim=$num_targets learning-rate-factor=$learning_rate_factor max-change=1.5 $output_opts
EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
fi


if [ $stage -le 6 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{3,4,5,6}/$USER/kaldi-data/egs/iam-$(date +'%m_%d_%H_%M')/s5/$dir/egs/storage $dir/egs/storage
  fi

  steps/nnet3/chain/train.py --stage=$train_stage \
    --cmd="$cmd" \
    --feat.cmvn-opts="--norm-means=false --norm-vars=false" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient=0.1 \
    --chain.apply-deriv-weights=true \
    --chain.lm-opts="--ngram-order=2 --no-prune-ngram-order=1 --num-extra-lm-states=1000" \
    --chain.frame-subsampling-factor=4 \
    --chain.alignment-subsampling-factor=1 \
    --chain.left-tolerance 3 \
    --chain.right-tolerance 3 \
    --trainer.srand=$srand \
    --trainer.max-param-change=2.0 \
    --trainer.num-epochs=10 \
    --trainer.frames-per-iter=2000000 \
    --trainer.optimization.num-jobs-initial=3 \
    --trainer.optimization.num-jobs-final=8 \
    --trainer.dropout-schedule $dropout_schedule \
    --trainer.optimization.initial-effective-lrate=0.001 \
    --trainer.optimization.final-effective-lrate=0.0001 \
    --trainer.optimization.shrink-value=1.0 \
    --trainer.num-chunk-per-minibatch=32,16 \
    --trainer.optimization.momentum=0.0 \
    --egs.chunk-width=$chunk_width \
    --egs.dir="$common_egs_dir" \
    --egs.opts="--frames-overlap-per-eg 0 --constrained false" \
    --cleanup.remove-egs=$remove_egs \
    --use-gpu=true \
    --reporting.email="$reporting_email" \
    --feat-dir=$train_data_dir \
    --tree-dir=$tree_dir \
    --lat-dir=$lat_dir \
    --dir=$dir  || exit 1;
fi

if [ $stage -le 7 ]; then
  # The reason we are using data/lang here, instead of $lang, is just to
  # emphasize that it's not actually important to give mkgraph.sh the
  # lang directory with the matched topology (since it gets the
  # topology file from the model).  So you could give it a different
  # lang directory, one that contained a wordlist and LM of your choice,
  # as long as phones.txt was compatible.

  utils/mkgraph.sh \
    --self-loop-scale 1.0 $lang_decode \
    $dir $dir/graph || exit 1;
fi

if [ $stage -le 8 ]; then
  frames_per_chunk=$(echo $chunk_width | cut -d, -f1)
  for decode_set in test $maybe_val; do
    steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
      --frames-per-chunk $frames_per_chunk \
      --nj $nj --cmd "$cmd" \
      $dir/graph data/$decode_set $dir/decode_$decode_set || exit 1;
  done
fi

echo "Done. Date: $(date). Results:"
local/chain/compare_wer.sh $dir
