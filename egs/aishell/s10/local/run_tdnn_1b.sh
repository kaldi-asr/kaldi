#!/bin/bash

# This script is based on run_tdnn_7h.sh in swbd chain recipe.

set -e

nj=10
# configs for 'chain'
affix=
stage=0
train_stage=-10
get_egs_stage=-10
dir=exp/chain_nnet3/tdnn_1b
decode_iter=

# training options
num_epochs=6
initial_effective_lrate=0.001
final_effective_lrate=0.0001
max_param_change=2.0
final_layer_normalize_target=0.5
num_jobs_initial=2
num_jobs_final=12
minibatch_size=128
frames_per_eg=150,110,90
remove_egs=true
common_egs_dir=
xent_regularize=0.1

# End configuration section.
echo "$0 $@"  # Print the command line for logging

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

if ! cuda-compiled; then
  echo "This script is intended to be used with GPUs"
  echo "but you have not compiled Kaldi with CUDA"
  echo "If you want to use GPUs (and have them), go to src/,"
  echo "and configure and make on a machine where "nvcc" is installed."
  exit 1
fi

dir=${dir}${affix:+_$affix}
train_set=train
ali_dir=exp/tri5a_ali
lat_dir=exp/tri5a_lats
treedir=exp/chain_nnet3/tri5_tree
lang=data/lang_chain_nnet3


if [[ $stage -le 0 ]]; then
  for datadir in train dev test; do
    dst_dir=data/fbank_pitch/$datadir
    if [[ ! -f $dst_dir/feats.scp ]]; then
      utils/copy_data_dir.sh data/$datadir $dst_dir
      echo "making fbank-pitch features for LF-MMI training"
      steps/make_fbank_pitch.sh --cmd $train_cmd --nj $nj $dst_dir || exit 1
      steps/compute_cmvn_stats.sh $dst_dir || exit 1
      utils/fix_data_dir.sh $dst_dir
    else
      echo "$dst_dir/feats.scp already exists."
      echo "kaldi pybind (local/run_chain.sh) LF-MMI may have generated it."
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
      --cmd $train_cmd 5000 data/train $lang $ali_dir $treedir
fi

if [[ $stage -le 3 ]]; then
  echo "creating neural net configs using the xconfig parser"

  num_targets=$(tree-info $treedir/tree | grep num-pdfs | awk '{print $2}')
  learning_rate_factor=$(echo "print(0.5/$xent_regularize)" | python3)
  feat_dim=$(feat-to-dim scp:data/fbank_pitch/train/feats.scp -)

  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig
  input dim=$feat_dim name=input

  # please note that it is important to have input layer with the name=input
  # as the layer immediately preceding the fixed-affine-layer to enable
  # the use of short notation for the descriptor
  fixed-affine-layer name=lda input=Append(-1,0,1) affine-transform-file=$dir/configs/lda.mat

  # the first splicing is moved before the lda layer, so no splicing here
  relu-batchnorm-layer name=tdnn1 dim=625
  relu-batchnorm-layer name=tdnn2 input=Append(-1,0,1) dim=625
  relu-batchnorm-layer name=tdnn3 input=Append(-1,0,1) dim=625
  relu-batchnorm-layer name=tdnn4 input=Append(-3,0,3) dim=625
  relu-batchnorm-layer name=tdnn5 input=Append(-3,0,3) dim=625
  relu-batchnorm-layer name=tdnn6 input=Append(-3,0,3) dim=625

  ## adding the layers for chain branch
  relu-batchnorm-layer name=prefinal-chain input=tdnn6 dim=625 target-rms=0.5
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
  relu-batchnorm-layer name=prefinal-xent input=tdnn6 dim=625 target-rms=0.5
  output-layer name=output-xent dim=$num_targets learning-rate-factor=$learning_rate_factor max-change=1.5

EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
fi

if [[ $stage -le 4 ]]; then
  steps/nnet3/chain/train.py --stage $train_stage \
    --cmd $cuda_cmd \
    --feat.cmvn-opts "--norm-means=false --norm-vars=false" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient 0.1 \
    --chain.l2-regularize 0.00005 \
    --chain.apply-deriv-weights false \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --egs.dir "$common_egs_dir" \
    --egs.stage $get_egs_stage \
    --egs.opts "--frames-overlap-per-eg 0" \
    --egs.chunk-width $frames_per_eg \
    --trainer.num-chunk-per-minibatch $minibatch_size \
    --trainer.frames-per-iter 1500000 \
    --trainer.num-epochs $num_epochs \
    --trainer.optimization.num-jobs-initial $num_jobs_initial \
    --trainer.optimization.num-jobs-final $num_jobs_final \
    --trainer.optimization.initial-effective-lrate $initial_effective_lrate \
    --trainer.optimization.final-effective-lrate $final_effective_lrate \
    --trainer.max-param-change $max_param_change \
    --cleanup.remove-egs $remove_egs \
		--cleanup.preserve-model-interval=1 \
    --feat-dir data/fbank_pitch/train \
    --tree-dir $treedir \
    --use-gpu "wait" \
    --lat-dir $lat_dir \
    --dir $dir  || exit 1
fi

if [[ $stage -le 5 ]]; then
  # Note: it might appear that this $lang directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.
  utils/mkgraph.sh --self-loop-scale 1.0 data/lang_test $dir $dir/graph
fi

graph_dir=$dir/graph
if [[ $stage -le 6 ]]; then
  for test_set in dev test; do
    steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
      --nj $nj --cmd $decode_cmd \
      $graph_dir data/fbank_pitch/${test_set} $dir/decode_${test_set} || exit 1
  done
fi
