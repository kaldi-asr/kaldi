#!/bin/bash

# this script is a modified version of swbd/run_tdnn_5f.sh

set -e

# configs for 'chain'
stage=0
train_stage=-10
get_egs_stage=-10
dir=exp/chain/tdnn_5f

# training options
num_epochs=12
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

# End configuration section.
echo "$0 $@"  # Print the command line for logging

. cmd.sh
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
# nnet2 setup, and you can skip them by setting "--stage 4" if you have already
# run those things.

ali_dir=exp/tri3b_ali
treedir=exp/chain/tri4_2y_tree
lang=data/lang_chain_2y

local/online/run_nnet2_common.sh --stage $stage || exit 1;

if [ $stage -le 4 ]; then
  # Get the alignments as lattices (gives the CTC training more freedom).
  # use the same num-jobs as the alignments
  nj=$(cat exp/tri3b_ali/num_jobs) || exit 1;
  steps/align_fmllr_lats.sh --nj $nj --cmd "$train_cmd -l q=all.q" data/train \
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
    --cmd "$train_cmd -l q=all.q" 1200 data/train $lang $ali_dir $treedir
fi

if [ $stage -le 7 ]; then
  steps/nnet3/chain/train_tdnn.sh --stage $train_stage \
    --xent-regularize 0.1 \
    --leaky-hmm-coefficient 0.1 \
    --l2-regularize 0.00005 \
    --jesus-opts "--jesus-forward-input-dim 200  --jesus-forward-output-dim 500 --jesus-hidden-dim 2000 --jesus-stddev-scale 0.2 --final-layer-learning-rate-factor 0.25" \
    --splice-indexes "-1,0,1 -2,-1,0,1 -3,0,3 -6,-3,0" \
    --apply-deriv-weights false \
    --frames-per-iter 1000000 \
    --lm-opts "--num-extra-lm-states=200" \
    --get-egs-stage $get_egs_stage \
    --minibatch-size $minibatch_size \
    --egs-opts "--frames-overlap-per-eg 0" \
    --frames-per-eg $frames_per_eg \
    --num-epochs $num_epochs --num-jobs-initial $num_jobs_initial --num-jobs-final $num_jobs_final \
    --feat-type raw \
    --online-ivector-dir exp/nnet2_online/ivectors \
    --cmvn-opts "--norm-means=false --norm-vars=false" \
    --initial-effective-lrate $initial_effective_lrate --final-effective-lrate $final_effective_lrate \
    --max-param-change $max_param_change \
    --cmd "$decode_cmd" \
    --remove-egs $remove_egs \
    data/train $treedir exp/tri3b_lats $dir  || exit 1;
fi

if [ $stage -le 8 ]; then
  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 4 \
    data/test exp/nnet2_online/extractor exp/nnet2_online/ivectors_test || exit 1;
fi

if [ $stage -le 9 ]; then
  # Note: it might appear that this $lang directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.
  utils/mkgraph.sh --self-loop-scale 1.0 data/lang $dir $dir/graph
  steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
    --extra-left-context 20 --scoring-opts "--min-lmwt 1" \
    --nj 20 --cmd "$decode_cmd" \
    --online-ivector-dir exp/nnet2_online/ivectors_test \
    $dir/graph data/test $dir/decode || exit 1;
fi

if [ $stage -le 10 ]; then
  utils/mkgraph.sh --self-loop-scale 1.0 data/lang_ug $dir $dir/graph_ug
  steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
    --extra-left-context 20 \
    --nj 20 --cmd "$decode_cmd" \
    --online-ivector-dir exp/nnet2_online/ivectors_test \
    $dir/graph_ug data/test $dir/decode_ug || exit 1;
fi
wait;
exit 0;
