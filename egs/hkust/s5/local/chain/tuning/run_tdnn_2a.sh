#!/bin/bash

# This script is based on run_tdnn_7p.sh in swbd chain recipe.

# Results
# local/chain/compare_wer.sh --online exp/chain/tdnn_7h_chain_2b_sp
# Model                tdnn_7h_chain_2b_sp
# CER(%)                    23.67
# CER(%)[online]            23.69
# CER(%)[per-utt]           24.67
# Final train prob        -0.0895
# Final valid prob        -0.1251
# Final train prob (xent)   -1.3628
# Final valid prob (xent)   -1.5590

# exp 2b: changes on network arch with multiple training options, referencing swbd
set -euxo pipefail

# configs for 'chain'
affix=chain_2a
stage=12
nj=10
train_stage=-10
get_egs_stage=-10
dir=exp/chain/tdnn_7h  # Note: _sp will get added to this if $speed_perturb == true.
decode_iter=

# training options
num_epochs=4
initial_effective_lrate=0.0005
final_effective_lrate=0.00005
max_param_change=2.0
final_layer_normalize_target=0.5
num_jobs_initial=3
num_jobs_final=3
minibatch_size=128
frames_per_eg=150,110,100
remove_egs=false
common_egs_dir=
xent_regularize=0.1
dropout_schedule='0,0@0.20,0.5@0.50,0'

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
# nnet3 setup, and you can skip them by setting "--stage 8" if you have already
# run those things.

dir=${dir}${affix:+_$affix}_sp
train_set=train_sp
ali_dir=exp/tri5a_sp_ali
treedir=exp/chain/tri6_7d_tree_sp
lang=data/lang_chain


# if we are using the speed-perturbed data we need to generate
# alignments for it.
if [ $stage -le 8 ]; then
  local/nnet3/run_ivector_common.sh --stage $stage \
    --ivector-extractor exp/nnet3/extractor || exit 1;
fi

if [ $stage -le 9 ]; then
  # Get the alignments as lattices (gives the LF-MMI training more freedom).
  # use the same num-jobs as the alignments
  nj=$(cat $ali_dir/num_jobs) || exit 1;
  steps/align_fmllr_lats.sh --nj $nj --cmd "$train_cmd" data/$train_set \
    data/lang exp/tri5a exp/tri5a_sp_lats
  rm exp/tri5a_sp_lats/fsts.*.gz # save space
fi

if [ $stage -le 10 ]; then
  # Create a version of the lang/ directory that has one state per phone in the
  # topo file. [note, it really has two states.. the first one is only repeated
  # once, the second one has zero or more repeats.]
  rm -rf $lang
  cp -r data/lang $lang
  silphonelist=$(cat $lang/phones/silence.csl) || exit 1;
  nonsilphonelist=$(cat $lang/phones/nonsilence.csl) || exit 1;
  # Use our special topology... note that later on may have to tune this
  # topology.
  steps/nnet3/chain/gen_topo.py $nonsilphonelist $silphonelist >$lang/topo
fi

if [ $stage -le 11 ]; then
  # Build a tree using our new topology. This is the critically different
  # step compared with other recipes.
  steps/nnet3/chain/build_tree.sh --frame-subsampling-factor 3 \
      --context-opts "--context-width=2 --central-position=1" \
      --cmd "$train_cmd" 5000 data/$train_set $lang $ali_dir $treedir
fi

if [ $stage -le 12 ]; then
  echo "$0: creating neural net configs using the xconfig parser";

  ivector_dim=$(feat-to-dim scp:exp/nnet3/ivectors_${train_set}/ivector_online.scp -)
  feat_dim=$(feat-to-dim scp:data/${train_set}_hires/feats.scp -)
  num_targets=$(tree-info $treedir/tree |grep num-pdfs|awk '{print $2}')
  learning_rate_factor=$(echo "print 0.5/$xent_regularize" | python)
  opts="l2-regularize=0.004 dropout-proportion=0.0 dropout-per-dim=true dropout-per-dim-continuous=true"
  linear_opts="orthonormal-constraint=-1.0 l2-regularize=0.004"
  output_opts="l2-regularize=0.002"

  mkdir -p $dir/configs

  cat <<EOF > $dir/configs/network.xconfig
  input dim=$ivector_dim name=ivector
  input dim=$feat_dim name=input
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

if [ $stage -le 13 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{5,6,7,8}/$USER/kaldi-data/egs/hkust-$(date +'%m_%d_%H_%M')/s5c/$dir/egs/storage $dir/egs/storage
  fi

  steps/nnet3/chain/train.py --stage $train_stage \
    --cmd "$decode_cmd" \
    --feat.online-ivector-dir exp/nnet3/ivectors_${train_set} \
    --feat.cmvn-opts "--norm-means=false --norm-vars=false" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient 0.1 \
    --chain.l2-regularize 0.0 \
    --chain.apply-deriv-weights false \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --trainer.dropout-schedule $dropout_schedule \
    --trainer.add-option="--optimization.memory-compression-level=2" \
    --egs.dir "$common_egs_dir" \
    --egs.stage $get_egs_stage \
    --egs.opts "--frames-overlap-per-eg 0 --constrained false" \
    --egs.chunk-width $frames_per_eg \
    --trainer.num-chunk-per-minibatch $minibatch_size \
    --trainer.optimization.momentum 0.0 \
    --trainer.frames-per-iter 1500000 \
    --trainer.num-epochs $num_epochs \
    --trainer.optimization.num-jobs-initial $num_jobs_initial \
    --trainer.optimization.num-jobs-final $num_jobs_final \
    --trainer.optimization.initial-effective-lrate $initial_effective_lrate \
    --trainer.optimization.final-effective-lrate $final_effective_lrate \
    --trainer.max-param-change $max_param_change \
    --cleanup.remove-egs $remove_egs \
    --feat-dir data/${train_set}_hires \
    --tree-dir $treedir \
    --lat-dir exp/tri5a_sp_lats \
    --dir $dir  || exit 1;
fi

if [ $stage -le 14 ]; then
  # Note: it might appear that this $lang directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.
  utils/mkgraph.sh --self-loop-scale 1.0 data/lang_test $dir $dir/graph
fi

graph_dir=$dir/graph
if [ $stage -le 15 ]; then
  iter_opts=
  if [ ! -z $decode_iter ]; then
    iter_opts=" --iter $decode_iter "
  fi
  steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
      --nj $nj --cmd "$decode_cmd" $iter_opts \
      --online-ivector-dir exp/nnet3/ivectors_dev \
    $graph_dir data/dev_hires $dir/decode || exit 1;
fi

if [ $stage -le 16 ]; then
  steps/online/nnet3/prepare_online_decoding.sh --mfcc-config conf/mfcc_hires.conf \
    --add-pitch true \
    data/lang exp/nnet3/extractor "$dir" ${dir}_online || exit 1;
fi

if [ $stage -le 17 ]; then
  # do the actual online decoding with iVectors, carrying info forward from
  # previous utterances of the same speaker.
  steps/online/nnet3/decode.sh --config conf/decode.config \
    --cmd "$decode_cmd" --nj $nj --acwt 1.0 --post-decode-acwt 10.0 \
    "$graph_dir" data/dev_hires \
    ${dir}_online/decode || exit 1;
fi

if [ $stage -le 18 ]; then
  # this version of the decoding treats each utterance separately
  # without carrying forward speaker information.
  steps/online/nnet3/decode.sh --config conf/decode.config \
    --cmd "$decode_cmd" --nj $nj --per-utt true --acwt 1.0 --post-decode-acwt 10.0 \
    "$graph_dir" data/dev_hires \
    ${dir}_online/decode_per_utt || exit 1;
fi
