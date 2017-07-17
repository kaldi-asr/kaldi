#!/bin/bash

# This script uses weight transfer as a Transfer learning method
# and use already trained model on wsj and removes the last layer and
# add new randomly initialized layer and retrain the whole network,
# while training new added layer using rm data.
# The chain config is as in run_tdnn_5n.sh and the result is:
#System tdnn_5n tdnn_wsj_rm_1a
#WER      2.71     2.09
set -e

# configs for 'chain'
stage=0
train_stage=-10
get_egs_stage=-10
dir=exp/chain/tdnn_wsj_rm_1a

# training options
num_epochs=2
initial_effective_lrate=0.005
final_effective_lrate=0.0005
leftmost_questions_truncate=-1
max_param_change=2.0
final_layer_normalize_target=0.5
num_jobs_initial=2
num_jobs_final=4
minibatch_size=128
frames_per_eg=150
remove_egs=false
xent_regularize=0.1

# configs for transfer learning
srcdir=../../wsj/s5/
common_egs_dir=exp/chain/tdnn_wsj_rm_1c_fixed_ac_scale/egs
src_mdl=$srcdir/exp/chain/tdnn1d_sp/final.mdl
primary_lr_factor=0.25
dim=450
nnet_affix=_online
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

ali_dir=exp/tri3b_ali
treedir=exp/chain/tri4_5n_tree
lang=data/lang_chain_5n

local/online/run_nnet2_common.sh  --stage $stage \
                                  --ivector-dim 100 \
                                  --nnet-affix "$nnet_affix" \
                                  --mfcc-config $srcdir/conf/mfcc_hires.conf \
                                  --extractor $srcdir/exp/nnet3/extractor || exit 1;

if [ $stage -le 4 ]; then
  # Get the alignments as lattices (gives the chain training more freedom).
  # use the same num-jobs as the alignments
  nj=$(cat exp/tri3b_ali/num_jobs) || exit 1;
  steps/align_fmllr_lats.sh --nj $nj --cmd "$train_cmd" data/train \
    data/lang exp/tri3b exp/tri3b_lats
  rm exp/tri3b_lats/fsts.*.gz # save space
fi

if [ $stage -le 5 ]; then
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

if [ $stage -le 6 ]; then
  # Build a tree using our new topology.
  steps/nnet3/chain/build_tree.sh --frame-subsampling-factor 3 \
    --leftmost-questions-truncate $leftmost_questions_truncate \
    --cmd "$train_cmd" 1200 data/train $lang $ali_dir $treedir
fi

if [ $stage -le 7 ]; then
  echo "$0: creating neural net configs using the xconfig parser for";
  echo "extra layers w.r.t source network.";
  num_targets=$(tree-info $treedir/tree |grep num-pdfs|awk '{print $2}')
  learning_rate_factor=$(echo "print 0.5/$xent_regularize" | python)
  mkdir -p $dir
  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig
  relu-renorm-layer name=tdnn7-target input=Append(tdnn6.renorm@-3,tdnn6.renorm@0) dim=$dim
  ## adding the layers for chain branch
  relu-renorm-layer name=prefinal-chain-target input=tdnn7-target dim=$dim target-rms=0.5
  output-layer name=output-target include-log-softmax=false dim=$num_targets max-change=1.5
  relu-renorm-layer name=prefinal-xent-target input=tdnn7-target dim=$dim target-rms=0.5
  output-layer name=output-xent-target dim=$num_targets learning-rate-factor=$learning_rate_factor max-change=1.5
EOF
  # edits.config contains edits required to train transferred model.
  # e.g. substitute output-node of previous model with new output
  # and removing orphan nodes and components.
  cat <<EOF > $dir/configs/edits.config
  remove-output-nodes name=output
  remove-output-nodes name=output-xent
  rename-node old-name=output-target new-name=output
  rename-node old-name=output-xent-target new-name=output-xent
  remove-orphans
EOF
  steps/nnet3/xconfig_to_configs.py --existing-model $src_mdl \
    --xconfig-file  $dir/configs/network.xconfig  \
    --edits-config $dir/configs/edits.config \
    --config-dir $dir/configs/
fi

if [ $stage -le 8 ]; then
  echo "$0: generate egs for chain to train new model on rm dataset."
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{3,4,5,6}/$USER/kaldi-data/egs/rm-$(date +'%m_%d_%H_%M')/s5c/$dir/egs/storage $dir/egs/storage
  fi
  echo "$0: set the learning-rate-factor for initial network to be zero."
  nnet3-copy --edits="set-learning-rate-factor name=* learning-rate-factor=$primary_lr_factor" \
    $src_mdl $dir/init.raw || exit 1;

  steps/nnet3/chain/train.py --stage $train_stage \
    --cmd "$decode_cmd" \
    --feat.online-ivector-dir exp/nnet2${nnet_affix}/ivectors \
    --chain.xent-regularize $xent_regularize \
    --feat.cmvn-opts "--norm-means=false --norm-vars=false" \
    --chain.xent-regularize 0.1 \
    --chain.leaky-hmm-coefficient 0.1 \
    --chain.l2-regularize 0.00005 \
    --chain.apply-deriv-weights false \
    --chain.lm-opts="--num-extra-lm-states=200" \
    --egs.dir "$common_egs_dir" \
    --egs.opts "--frames-overlap-per-eg 0" \
    --egs.chunk-width $frames_per_eg \
    --trainer.num-chunk-per-minibatch=$minibatch_size \
    --trainer.frames-per-iter 1000000 \
    --trainer.num-epochs $num_epochs \
    --trainer.optimization.num-jobs-initial=$num_jobs_initial \
    --trainer.optimization.num-jobs-final=$num_jobs_final \
    --trainer.optimization.initial-effective-lrate=$initial_effective_lrate \
    --trainer.optimization.final-effective-lrate=$final_effective_lrate \
    --trainer.max-param-change $max_param_change \
    --cleanup.remove-egs false \
    --feat-dir data/train_hires \
    --tree-dir $treedir \
    --lat-dir exp/tri3b_lats \
    --dir $dir || exit 1;
fi

if [ $stage -le 9 ]; then
  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 4 \
    data/test $srcdir/exp/nnet3/extractor exp/nnet2${nnet_affix}/ivectors_test || exit 1;
fi

if [ $stage -le 10 ]; then
  # Note: it might appear that this $lang directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.
  utils/mkgraph.sh --self-loop-scale 1.0 data/lang $dir $dir/graph
  steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
    --scoring-opts "--min-lmwt 1" \
    --nj 20 --cmd "$decode_cmd" \
    --online-ivector-dir exp/nnet2${nnet_affix}/ivectors_test \
    $dir/graph data/test_hires $dir/decode || exit 1;
fi

if [ $stage -le 11 ]; then
  utils/mkgraph.sh --self-loop-scale 1.0 data/lang_ug $dir $dir/graph_ug
  steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
    --nj 20 --cmd "$decode_cmd" \
    --online-ivector-dir exp/nnet2${nnet_affix}/ivectors_test \
    $dir/graph_ug data/test_hires $dir/decode_ug || exit 1;
fi
wait;
exit 0;
