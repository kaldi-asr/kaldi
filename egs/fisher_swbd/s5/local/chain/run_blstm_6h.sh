#!/usr/bin/env bash

# based on run_tdnn_6h.sh

set -e

# configs for 'chain'
stage=12
train_stage=-10
get_egs_stage=-10
dir=exp/chain/blstm_6h
decode_iter=
decode_dir_affix=

# training options
num_epochs=4
remove_egs=false
common_egs_dir=
affix=
chunk_width=150
chunk_left_context=40
chunk_right_context=40

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

dir=$dir${affix:+_$affix}
build_tree_train_set=train_nodup
train_set=train_nodup_sp
build_tree_ali_dir=exp/tri5a_ali
treedir=exp/chain/tri6_tree
lang=data/lang_chain

# The iVector-extraction and feature-dumping parts are the same as the standard
# nnet3 setup, and you can skip them by setting "--stage 8" if you have already
# run those things.
local/nnet3/run_ivector_common.sh --stage $stage \
  --speed-perturb true \
  --generate-alignments false || exit 1;

if [ $stage -le 9 ]; then
  # Get the alignments as lattices (gives the CTC training more freedom).
  # use the same num-jobs as the alignments
  nj=$(cat $build_tree_ali_dir/num_jobs) || exit 1;
  steps/align_fmllr_lats.sh --nj $nj --cmd "$train_cmd" data/$train_set \
    data/lang exp/tri5a exp/tri5a_lats_nodup_sp || exit 1;
  rm exp/tri5a_lats_nodup_sp/fsts.*.gz # save space
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
  # Build a tree using our new topology.
  steps/nnet3/chain/build_tree.sh --frame-subsampling-factor 3 \
      --cmd "$train_cmd" 11000 data/$build_tree_train_set $lang $build_tree_ali_dir $treedir || exit 1;
fi

if [ $stage -le 12 ]; then
  echo "$0: creating neural net configs";

  steps/nnet3/lstm/make_configs.py  \
    --feat-dir data/${train_set}_hires \
    --ivector-dir exp/nnet3/ivectors_${train_set} \
    --tree-dir $treedir \
    --splice-indexes="-2,-1,0,1,2 0 0" \
    --lstm-delay=" [-3,3] [-3,3] [-3,3] " \
    --xent-regularize 0.1 \
    --include-log-softmax false \
    --num-lstm-layers 3 \
    --cell-dim 1024 \
    --hidden-dim 1024 \
    --recurrent-projection-dim 256 \
    --non-recurrent-projection-dim 256 \
    --label-delay 0 \
    --self-repair-scale 0.00001 \
   $dir/configs || exit 1;

fi

if [ $stage -le 13 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{5,6,7,8}/$USER/kaldi-data/egs/fisher_swbd-$(date +'%m_%d_%H_%M')/s5c/$dir/egs/storage $dir/egs/storage
  fi

  steps/nnet3/chain/train.py --stage $train_stage \
    --cmd "$decode_cmd" \
    --feat.online-ivector-dir exp/nnet3/ivectors_${train_set} \
    --feat.cmvn-opts "--norm-means=false --norm-vars=false" \
    --chain.xent-regularize 0.1 \
    --chain.leaky-hmm-coefficient 0.1 \
    --chain.l2-regularize 0.00005 \
    --chain.apply-deriv-weights false \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --trainer.num-chunk-per-minibatch 64 \
    --trainer.frames-per-iter 1200000 \
    --trainer.max-param-change 1.414 \
    --trainer.num-epochs $num_epochs \
    --trainer.optimization.shrink-value 0.99 \
    --trainer.optimization.num-jobs-initial 3 \
    --trainer.optimization.num-jobs-final 16 \
    --trainer.optimization.initial-effective-lrate 0.001 \
    --trainer.optimization.final-effective-lrate 0.0001 \
    --trainer.optimization.momentum 0.0 \
    --trainer.deriv-truncate-margin 8 \
    --egs.stage $get_egs_stage \
    --egs.opts "--frames-overlap-per-eg 0" \
    --egs.chunk-width $chunk_width \
    --egs.chunk-left-context $chunk_left_context \
    --egs.chunk-right-context $chunk_right_context \
    --egs.dir "$common_egs_dir" \
    --cleanup.remove-egs $remove_egs \
    --feat-dir data/${train_set}_hires \
    --tree-dir $treedir \
    --lat-dir exp/tri5a_lats_nodup_sp \
    --dir $dir  || exit 1;
fi

if [ $stage -le 14 ]; then
  # Note: it might appear that this $lang directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.
  utils/mkgraph.sh --self-loop-scale 1.0 data/lang_fsh_sw1_tg $dir $dir/graph_fsh_sw1_tg
fi

decode_suff=fsh_sw1_tg
graph_dir=$dir/graph_fsh_sw1_tg
if [ $stage -le 15 ]; then
  iter_opts=
  if [ ! -z $decode_iter ]; then
    iter_opts=" --iter $decode_iter "
  fi

  # decoding options
  extra_left_context=$[$chunk_left_context+10]
  extra_right_context=$[$chunk_right_context+10]

  for decode_set in eval2000 rt03; do
      (
      num_jobs=`cat data/${decode_set}_hires/utt2spk|cut -d' ' -f2|sort -u|wc -l`
      steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
          --nj $num_jobs --cmd "$decode_cmd" $iter_opts \
          --extra-left-context $extra_left_context \
          --extra-right-context $extra_right_context \
          --frames-per-chunk $chunk_width \
          --online-ivector-dir exp/nnet3/ivectors_${decode_set} \
         $graph_dir data/${decode_set}_hires $dir/decode_${decode_set}${decode_dir_affix:+_$decode_dir_affix}_${decode_suff} || exit 1;
      steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
          data/lang_fsh_sw1_{tg,fg} data/${decode_set}_hires \
         $dir/decode_${decode_set}${decode_dir_affix:+_$decode_dir_affix}_fsh_sw1_{tg,fg} || exit 1;
      ) &
  done
fi
wait;
exit 0;
