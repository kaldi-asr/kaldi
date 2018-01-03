#!/bin/bash
# Copyright 2017 University of Chinese Academy of Sciences (UCAS) Gaofeng Cheng
# Apache 2.0

# The model training procedure is similar to run_blstm_6j.sh under egs/swbd/s5c

# ./local/chain/compare_wer_general.sh blstm_6j_sp
# System                 blstm_6j_sp
# WER on eval2000(tg)      12.1
# WER on eval2000(fg)      11.9
# WER on rt03(tg)          11.9
# WER on rt03(fg)          11.6
# Final train prob         -0.059
# Final valid prob         -0.072
# Final train prob (xent)        -0.711
# Final valid prob (xent)       -0.7782

# ./local/chain/show_chain_wer.sh blstm_6j_sp
# %WER 15.2 | 2628 21594 | 87.0 8.2 4.8 2.2 15.2 52.0 | exp/chain/blstm_6j_sp/decode_eval2000_fsh_sw1_tg/score_7_0.0/eval2000_hires.ctm.callhm.filt.sys
# %WER 12.1 | 4459 42989 | 89.8 6.8 3.4 1.9 12.1 49.4 | exp/chain/blstm_6j_sp/decode_eval2000_fsh_sw1_tg/score_7_0.0/eval2000_hires.ctm.filt.sys
# %WER 8.5 | 1831 21395 | 92.7 5.1 2.1 1.3 8.5 44.0 | exp/chain/blstm_6j_sp/decode_eval2000_fsh_sw1_tg/score_8_1.0/eval2000_hires.ctm.swbd.filt.sys
# %WER 15.0 | 2628 21594 | 87.2 8.1 4.7 2.2 15.0 51.4 | exp/chain/blstm_6j_sp/decode_eval2000_fsh_sw1_fg/score_7_0.0/eval2000_hires.ctm.callhm.filt.sys
# %WER 11.9 | 4459 42989 | 90.0 6.7 3.3 1.8 11.9 48.6 | exp/chain/blstm_6j_sp/decode_eval2000_fsh_sw1_fg/score_7_0.0/eval2000_hires.ctm.filt.sys
# %WER 8.5 | 1831 21395 | 92.7 5.0 2.3 1.2 8.5 43.7 | exp/chain/blstm_6j_sp/decode_eval2000_fsh_sw1_fg/score_10_0.0/eval2000_hires.ctm.swbd.filt.sys

# ./local/chain/show_chain_rt03_wer.sh blstm_6j_sp
# %WER 10.1 | 3970 36721 | 91.1 5.3 3.6 1.2 10.1 43.8 | exp/chain/blstm_6j_sp/decode_rt03_fsh_sw1_tg/score_7_0.0/rt03_hires.ctm.fsh.filt.sys
# %WER 11.9 | 8420 76157 | 89.6 6.6 3.8 1.5 11.9 45.2 | exp/chain/blstm_6j_sp/decode_rt03_fsh_sw1_tg/score_7_0.0/rt03_hires.ctm.filt.sys
# %WER 13.5 | 4450 39436 | 88.2 7.9 3.9 1.8 13.5 46.4 | exp/chain/blstm_6j_sp/decode_rt03_fsh_sw1_tg/score_7_0.0/rt03_hires.ctm.swbd.filt.sys
# %WER 9.7 | 3970 36721 | 91.5 5.1 3.5 1.2 9.7 43.4 | exp/chain/blstm_6j_sp/decode_rt03_fsh_sw1_fg/score_7_0.0/rt03_hires.ctm.fsh.filt.sys
# %WER 11.6 | 8420 76157 | 89.9 6.5 3.6 1.5 11.6 44.7 | exp/chain/blstm_6j_sp/decode_rt03_fsh_sw1_fg/score_7_0.0/rt03_hires.ctm.filt.sys
# %WER 13.3 | 4450 39436 | 88.5 7.7 3.8 1.8 13.3 45.8 | exp/chain/blstm_6j_sp/decode_rt03_fsh_sw1_fg/score_7_0.0/rt03_hires.ctm.swbd.filt.sys


set -e

# configs for 'chain'
stage=12
train_stage=-10
get_egs_stage=-10
dir=exp/chain/blstm_6j
decode_iter=
decode_dir_affix=

# training options
# training options
leftmost_questions_truncate=-1
chunk_width=150
chunk_left_context=40
chunk_right_context=40
xent_regularize=0.025
self_repair_scale=0.00001
label_delay=0

# decode options
extra_left_context=50
extra_right_context=50
frames_per_chunk=

remove_egs=false
common_egs_dir=

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

suffix=
if [ "$speed_perturb" == "true" ]; then
  suffix=_sp
fi

dir=${dir}$suffix
build_tree_train_set=train_nodup
train_set=train_nodup_sp
build_tree_ali_dir=exp/tri5a_ali
treedir=exp/chain/tri6_tree
lang=data/lang_chain

# if we are using the speed-perturbed data we need to generate
# alignments for it.
local/nnet3/run_ivector_common.sh --stage $stage \
  --speed-perturb $speed_perturb \
  --generate-alignments $speed_perturb || exit 1;

if [ $stage -le 9 ]; then
  # Get the alignments as lattices (gives the CTC training more freedom).
  # use the same num-jobs as the alignments
  nj=$(cat $build_tree_ali_dir/num_jobs) || exit 1;
  steps/align_fmllr_lats.sh --nj $nj --cmd "$train_cmd" data/$train_set \
    data/lang exp/tri5a exp/tri5a_lats_nodup$suffix || exit 1;
  rm exp/tri5a_lats_nodup$suffix/fsts.*.gz # save space
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
      --leftmost-questions-truncate $leftmost_questions_truncate \
      --context-opts "--context-width=2 --central-position=1" \
      --cmd "$train_cmd" 11000 data/$build_tree_train_set $lang $build_tree_ali_dir $treedir || exit 1
fi

if [ $stage -le 12 ]; then
  echo "$0: creating neural net configs using the xconfig parser";

  num_targets=$(tree-info $treedir/tree |grep num-pdfs|awk '{print $2}')
  learning_rate_factor=$(echo "print 0.5/$xent_regularize" | python)

  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig
  input dim=100 name=ivector
  input dim=40 name=input

  # please note that it is important to have input layer with the name=input
  # as the layer immediately preceding the fixed-affine-layer to enable
  # the use of short notation for the descriptor
  fixed-affine-layer name=lda input=Append(-2,-1,0,1,2,ReplaceIndex(ivector, t, 0)) affine-transform-file=$dir/configs/lda.mat

  # check steps/libs/nnet3/xconfig/lstm.py for the other options and defaults
  lstmp-layer name=blstm1-forward input=lda cell-dim=1024 recurrent-projection-dim=256 non-recurrent-projection-dim=256 delay=-3
  lstmp-layer name=blstm1-backward input=lda cell-dim=1024 recurrent-projection-dim=256 non-recurrent-projection-dim=256 delay=3

  lstmp-layer name=blstm2-forward input=Append(blstm1-forward, blstm1-backward) cell-dim=1024 recurrent-projection-dim=256 non-recurrent-projection-dim=256 delay=-3
  lstmp-layer name=blstm2-backward input=Append(blstm1-forward, blstm1-backward) cell-dim=1024 recurrent-projection-dim=256 non-recurrent-projection-dim=256 delay=3

  lstmp-layer name=blstm3-forward input=Append(blstm2-forward, blstm2-backward) cell-dim=1024 recurrent-projection-dim=256 non-recurrent-projection-dim=256 delay=-3
  lstmp-layer name=blstm3-backward input=Append(blstm2-forward, blstm2-backward) cell-dim=1024 recurrent-projection-dim=256 non-recurrent-projection-dim=256 delay=3

  ## adding the layers for chain branch
  output-layer name=output input=Append(blstm3-forward, blstm3-backward) output-delay=$label_delay include-log-softmax=false dim=$num_targets max-change=1.5

  # adding the layers for xent branch
  # This block prints the configs for a separate output that will be
  # trained with a cross-entropy objective in the 'chain' models... this
  # has the effect of regularizing the hidden parts of the model.  we use
  # 0.5 / args.xent_regularize as the learning rate factor- the factor of
  # 0.5 / args.xent_regularize is suitable as it means the xent
  # final-layer learns at a rate independent of the regularization
  # constant; and the 0.5 was tuned so as to make the relative progress
  # similar in the xent and regular final layers.
  output-layer name=output-xent input=Append(blstm3-forward, blstm3-backward) output-delay=$label_delay dim=$num_targets learning-rate-factor=$learning_rate_factor max-change=1.5

EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
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
    --trainer.max-param-change 2.0 \
    --trainer.num-epochs 4 \
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
    --lat-dir exp/tri5a_lats_nodup$suffix \
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
      fi
      ) &
  done
fi
wait;
exit 0;
