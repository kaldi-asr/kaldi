#!/bin/bash
# Copyright 2017 University of Chinese Academy of Sciences (UCAS) Gaofeng Cheng
# Apache 2.0

# Based on tdnn_7n (from egs/swbd/s5c).
# With the semi-orthogonal matrix used and skip connections added (wait reference here).
# Difference between   tdnn_7c and tdnn_7d:
# skip connections     No   Yes
# semi-orthogonal matrix    No   Yes
# frames_per_eg        150  150,110,100
# l2 in output         Yes  No
# epochs               4    6
# l2 in TDNN layers    No   Yes

# System                  tdnn_7c_sp    tdnn_7d_sp
# WER on eval2000(tg)        13.5      12.8
# WER on eval2000(fg)        13.3      12.6
# WER on rt03(tg)        12.7      11.8
# WER on rt03(fg)        12.5      11.5
# Final train prob         -0.103    -0.112
# Final valid prob         -0.114    -0.107
# Final train prob (xent)        -1.159    -1.262
# Final valid prob (xent)       -1.2024   -1.2200
# Num-parameters               18781673  20170506


# %WER 16.4 | 2628 21594 | 85.8 9.5 4.6 2.2 16.4 55.1 | exp/chain/tdnn_7d_sp/decode_eval2000_fsh_sw1_tg/score_8_0.0/eval2000_hires.ctm.callhm.filt.sys
# %WER 12.8 | 4459 42989 | 89.0 7.7 3.3 1.8 12.8 50.4 | exp/chain/tdnn_7d_sp/decode_eval2000_fsh_sw1_tg/score_8_0.0/eval2000_hires.ctm.filt.sys
# %WER 9.1 | 1831 21395 | 92.0 5.6 2.3 1.1 9.1 43.4 | exp/chain/tdnn_7d_sp/decode_eval2000_fsh_sw1_tg/score_10_0.0/eval2000_hires.ctm.swbd.filt.sys
# %WER 16.3 | 2628 21594 | 86.0 9.4 4.6 2.2 16.3 54.6 | exp/chain/tdnn_7d_sp/decode_eval2000_fsh_sw1_fg/score_8_0.0/eval2000_hires.ctm.callhm.filt.sys
# %WER 12.6 | 4459 42989 | 88.8 6.9 4.3 1.4 12.6 49.3 | exp/chain/tdnn_7d_sp/decode_eval2000_fsh_sw1_fg/score_10_0.0/eval2000_hires.ctm.filt.sys
# %WER 8.8 | 1831 21395 | 92.3 5.5 2.3 1.1 8.8 42.3 | exp/chain/tdnn_7d_sp/decode_eval2000_fsh_sw1_fg/score_10_1.0/eval2000_hires.ctm.swbd.filt.sys

# %WER 9.4 | 3970 36721 | 91.7 5.7 2.5 1.1 9.4 39.6 | exp/chain/tdnn_7d_sp/decode_rt03_fsh_sw1_tg/score_8_0.0/rt03_hires.ctm.fsh.filt.sys
# %WER 11.8 | 8420 76157 | 89.5 7.3 3.1 1.4 11.8 42.2 | exp/chain/tdnn_7d_sp/decode_rt03_fsh_sw1_tg/score_8_0.0/rt03_hires.ctm.filt.sys
# %WER 13.9 | 4450 39436 | 87.5 8.3 4.2 1.4 13.9 44.5 | exp/chain/tdnn_7d_sp/decode_rt03_fsh_sw1_tg/score_9_0.0/rt03_hires.ctm.swbd.filt.sys
# %WER 9.1 | 3970 36721 | 92.0 5.5 2.5 1.1 9.1 39.4 | exp/chain/tdnn_7d_sp/decode_rt03_fsh_sw1_fg/score_8_0.0/rt03_hires.ctm.fsh.filt.sys
# %WER 11.5 | 8420 76157 | 89.8 7.1 3.1 1.3 11.5 41.8 | exp/chain/tdnn_7d_sp/decode_rt03_fsh_sw1_fg/score_8_0.0/rt03_hires.ctm.filt.sys
# %WER 13.6 | 4450 39436 | 87.6 7.6 4.7 1.3 13.6 44.7 | exp/chain/tdnn_7d_sp/decode_rt03_fsh_sw1_fg/score_10_0.0/rt03_hires.ctm.swbd.filt.sys

set -e

# configs for 'chain'
stage=12
train_stage=-10
get_egs_stage=-10
speed_perturb=true
dir=exp/chain/tdnn_7d # Note: _sp will get added to this if $speed_perturb == true.
decode_iter=
decode_dir_affix=

# training options
leftmost_questions_truncate=-1
num_epochs=6
initial_effective_lrate=0.001
final_effective_lrate=0.0001
max_param_change=2.0
num_jobs_initial=3
num_jobs_final=16
minibatch_size=128
frames_per_eg=150,110,100
remove_egs=false
common_egs_dir=
xent_regularize=0.1

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
    data/lang exp/tri5a exp/tri5a_lats_nodup$suffix
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
      --cmd "$train_cmd" 11000 data/$build_tree_train_set $lang $build_tree_ali_dir $treedir
fi

if [ $stage -le 12 ]; then
  echo "$0: creating neural net configs using the xconfig parser";

  num_targets=$(tree-info $treedir/tree |grep num-pdfs|awk '{print $2}')
  learning_rate_factor=$(echo "print 0.5/$xent_regularize" | python)
  opts="l2-regularize=0.002"
  linear_opts="orthonormal-constraint=1.0"
  output_opts="l2-regularize=0.0005 bottleneck-dim=256"

  mkdir -p $dir/configs

  cat <<EOF > $dir/configs/network.xconfig
  input dim=100 name=ivector
  input dim=40 name=input

  # please note that it is important to have input layer with the name=input
  # as the layer immediately preceding the fixed-affine-layer to enable
  # the use of short notation for the descriptor
  fixed-affine-layer name=lda input=Append(-1,0,1,ReplaceIndex(ivector, t, 0)) affine-transform-file=$dir/configs/lda.mat

  # the first splicing is moved before the lda layer, so no splicing here
  relu-batchnorm-layer name=tdnn1 $opts dim=1280
  linear-component name=tdnn2l dim=256 $linear_opts input=Append(-1,0)
  relu-batchnorm-layer name=tdnn2 $opts input=Append(0,1) dim=1280
  linear-component name=tdnn3l dim=256 $linear_opts
  relu-batchnorm-layer name=tdnn3 $opts dim=1280
  linear-component name=tdnn4l dim=256 $linear_opts input=Append(-1,0)
  relu-batchnorm-layer name=tdnn4 $opts input=Append(0,1) dim=1280
  linear-component name=tdnn5l dim=256 $linear_opts
  relu-batchnorm-layer name=tdnn5 $opts dim=1280 input=Append(tdnn5l, tdnn3l)
  linear-component name=tdnn6l dim=256 $linear_opts input=Append(-3,0)
  relu-batchnorm-layer name=tdnn6 $opts input=Append(0,3) dim=1280
  linear-component name=tdnn7l dim=256 $linear_opts input=Append(-3,0)
  relu-batchnorm-layer name=tdnn7 $opts input=Append(0,3,tdnn6l,tdnn4l,tdnn2l) dim=1280
  linear-component name=tdnn8l dim=256 $linear_opts input=Append(-3,0)
  relu-batchnorm-layer name=tdnn8 $opts input=Append(0,3) dim=1280
  linear-component name=tdnn9l dim=256 $linear_opts input=Append(-3,0)
  relu-batchnorm-layer name=tdnn9 $opts input=Append(0,3,tdnn8l,tdnn6l,tdnn4l) dim=1280
  linear-component name=tdnn10l dim=256 $linear_opts input=Append(-3,0)
  relu-batchnorm-layer name=tdnn10 $opts input=Append(0,3) dim=1280
  linear-component name=tdnn11l dim=256 $linear_opts input=Append(-3,0)
  relu-batchnorm-layer name=tdnn11 $opts input=Append(0,3,tdnn10l,tdnn8l,tdnn6l) dim=1280
  linear-component name=prefinal-l dim=256 $linear_opts

  relu-batchnorm-layer name=prefinal-chain input=prefinal-l $opts dim=1280
  output-layer name=output include-log-softmax=false dim=$num_targets $output_opts

  relu-batchnorm-layer name=prefinal-xent input=prefinal-l $opts dim=1280
  output-layer name=output-xent dim=$num_targets learning-rate-factor=$learning_rate_factor $output_opts
EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
fi

if [ $stage -le 13 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{5,6,7,8}/$USER/kaldi-data/egs/swbd-$(date +'%m_%d_%H_%M')/s5c/$dir/egs/storage $dir/egs/storage
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
  rm $dir/.error 2>/dev/null || true
  if [ ! -z $decode_iter ]; then
    iter_opts=" --iter $decode_iter "
  fi
  for decode_set in rt03 eval2000; do
      (
      steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
          --nj 50 --cmd "$decode_cmd" $iter_opts \
          --online-ivector-dir exp/nnet3/ivectors_${decode_set} \
         $graph_dir data/${decode_set}_hires \
         $dir/decode_${decode_set}${decode_dir_affix:+_$decode_dir_affix}_${decode_suff} || exit 1;
      steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
            data/lang_fsh_sw1_{tg,fg} data/${decode_set}_hires \
            $dir/decode_${decode_set}${decode_dir_affix:+_$decode_dir_affix}_fsh_sw1_{tg,fg} || exit 1;
      ) || touch $dir/.error &
  done
  wait
  if [ -f $dir/.error ]; then
    echo "$0: something went wrong in decoding"
    exit 1
  fi
fi

test_online_decoding=true
lang=data/lang_fsh_sw1_tg
if $test_online_decoding && [ $stage -le 16 ]; then
  # note: if the features change (e.g. you add pitch features), you will have to
  # change the options of the following command line.
  steps/online/nnet3/prepare_online_decoding.sh \
       --mfcc-config conf/mfcc_hires.conf \
       $lang exp/nnet3/extractor $dir ${dir}_online

  rm $dir/.error 2>/dev/null || true
  for decode_set in rt03 eval2000; do
    (
      # note: we just give it "$decode_set" as it only uses the wav.scp, the
      # feature type does not matter.

      steps/online/nnet3/decode.sh --nj 50 --cmd "$decode_cmd" $iter_opts \
          --acwt 1.0 --post-decode-acwt 10.0 \
         $graph_dir data/${decode_set}_hires \
         ${dir}_online/decode_${decode_set}${decode_iter:+_$decode_iter}_${decode_suff} || exit 1;
	    steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
		      data/lang_fsh_sw1_{tg,fg} data/${decode_set}_hires \
		      ${dir}_online/decode_${decode_set}${decode_dir_affix:+_$decode_dir_affix}_fsh_sw1_{tg,fg} || exit 1;
    ) || touch $dir/.error &
  done
  wait
  if [ -f $dir/.error ]; then
    echo "$0: something went wrong in online decoding"
    exit 1
  fi
fi

exit 0;
