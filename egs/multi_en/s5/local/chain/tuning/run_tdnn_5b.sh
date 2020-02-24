#!/usr/bin/env bash
# Copyright 2018 Xiaohui Zhang
#           2017 University of Chinese Academy of Sciences (UCAS) Gaofeng Cheng
# Apache 2.0

# Based on tdnn_7o from egs/swbd/s5c, with larger network dimensions and different regularization coefficients.

# ./local/chain/compare_wer_general.sh tdnn_5b_sp
# System                tdnn_5b_sp
# WER on eval2000(tg)        11.7
# WER on eval2000(fg)        11.5
# WER on rt03(tg)            11.9
# WER on rt03(fg)            11.5
# Final train prob          -0.097
# Final valid prob          -0.090
# Final train prob (xent)   -1.042
# Final valid prob (xent)   -0.9712
# Num-parameters             34818416

# %WER 14.5 | 2628 21594 | 87.6 8.7 3.7 2.1 14.5 49.5 | exp/multi_a/chain/tdnn_5b_sp/decode_eval2000_fsh_sw1_tg/score_9_0.0/eval2000_hires.ctm.callhm.filt.sys
# %WER 11.7 | 4459 42989 | 90.0 7.3 2.7 1.7 11.7 47.1 | exp/multi_a/chain/tdnn_5b_sp/decode_eval2000_fsh_sw1_tg/score_9_0.0/eval2000_hires.ctm.filt.sys
# %WER 8.8 | 1831 21395 | 92.4 5.6 2.1 1.2 8.8 43.4 | exp/multi_a/chain/tdnn_5b_sp/decode_eval2000_fsh_sw1_tg/score_10_0.0/eval2000_hires.ctm.swbd.filt.sys
# %WER 14.4 | 2628 21594 | 87.8 9.0 3.2 2.2 14.4 49.7 | exp/multi_a/chain/tdnn_5b_sp/decode_eval2000_fsh_sw1_fg/score_8_0.5/eval2000_hires.ctm.callhm.filt.sys
# %WER 11.5 | 4459 42989 | 90.2 7.2 2.6 1.7 11.5 46.6 | exp/multi_a/chain/tdnn_5b_sp/decode_eval2000_fsh_sw1_fg/score_9_0.0/eval2000_hires.ctm.filt.sys
# %WER 8.6 | 1831 21395 | 92.6 5.5 1.9 1.2 8.6 42.6 | exp/multi_a/chain/tdnn_5b_sp/decode_eval2000_fsh_sw1_fg/score_10_0.0/eval2000_hires.ctm.swbd.filt.sys
# 
# %WER 10.1 | 3970 36721 | 91.3 6.0 2.8 1.4 10.1 44.0 | exp/multi_a/chain/tdnn_5b_sp/decode_rt03_fsh_sw1_tg/score_7_1.0/rt03_hires.ctm.fsh.filt.sys
# %WER 11.9 | 8420 76157 | 89.6 7.1 3.3 1.5 11.9 44.6 | exp/multi_a/chain/tdnn_5b_sp/decode_rt03_fsh_sw1_tg/score_8_0.0/rt03_hires.ctm.filt.sys
# %WER 13.5 | 4450 39436 | 88.0 8.0 4.0 1.5 13.5 44.8 | exp/multi_a/chain/tdnn_5b_sp/decode_rt03_fsh_sw1_tg/score_9_0.0/rt03_hires.ctm.swbd.filt.sys
# %WER 9.7 | 3970 36721 | 91.6 5.5 3.0 1.2 9.7 42.8 | exp/multi_a/chain/tdnn_5b_sp/decode_rt03_fsh_sw1_fg/score_8_0.0/rt03_hires.ctm.fsh.filt.sys
# %WER 11.5 | 8420 76157 | 89.8 6.5 3.7 1.3 11.5 43.9 | exp/multi_a/chain/tdnn_5b_sp/decode_rt03_fsh_sw1_fg/score_9_0.0/rt03_hires.ctm.filt.sys
# %WER 13.3 | 4450 39436 | 88.3 7.8 4.0 1.5 13.3 44.7 | exp/multi_a/chain/tdnn_5b_sp/decode_rt03_fsh_sw1_fg/score_9_0.0/rt03_hires.ctm.swbd.filt.sys

set -e

# configs for 'chain'
stage=1
train_stage=-10
get_egs_stage=-10
speed_perturb=true
multi=multi_a
gmm=tri5a
dir=exp/multi_a/chain/tdnn_5b # Note: _sp will get added to this if $speed_perturb == true.
decode_iter=
decode_dir_affix=
rescore=true # whether to rescore lattices

# training options
leftmost_questions_truncate=-1
num_epochs=3
initial_effective_lrate=0.0005
final_effective_lrate=0.00005
max_param_change=2.0
num_jobs_initial=3
num_jobs_final=16
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

suffix=
if [ "$speed_perturb" == "true" ]; then
  suffix=_sp
fi

dir=${dir}$suffix
train_set=multi_a/${gmm}_sp
build_tree_ali_dir=exp/multi_a/${gmm}_ali_sp
treedir=exp/multi_a/chain/${gmm}_tree
lang=data/multi_a/lang_${gmm}_chain

# if we are using the speed-perturbed data we need to generate
# alignments for it.
local/nnet3/run_ivector_common.sh --stage $stage \
  --multi multi_a \
  --gmm $gmm \
  --speed-perturb $speed_perturb \
  --generate-alignments $speed_perturb || exit 1;

if [ $stage -le 9 ]; then
  # Get the alignments as lattices (gives the LF-MMI training more freedom).
  # use the same num-jobs as the alignments
  nj=$(cat $build_tree_ali_dir/num_jobs) || exit 1;
  steps/align_fmllr_lats.sh --nj $nj --cmd "$train_cmd" data/$train_set \
    data/lang_${multi}_${gmm} exp/multi_a/$gmm exp/multi_a/${gmm}_lats_nodup$suffix
  rm exp/multi_a/${gmm}_lats_nodup$suffix/fsts.*.gz # save space
fi

if [ $stage -le 10 ]; then
  # Create a version of the lang/ directory that has one state per phone in the
  # topo file. [note, it really has two states.. the first one is only repeated
  # once, the second one has zero or more repeats.]
  rm -rf $lang
  cp -r data/lang_${multi}_${gmm} $lang
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
      --cmd "$train_cmd" 11000 data/$train_set $lang $build_tree_ali_dir $treedir
fi

if [ $stage -le 12 ]; then
  echo "$0: creating neural net configs using the xconfig parser";

  num_targets=$(tree-info $treedir/tree |grep num-pdfs|awk '{print $2}')
  learning_rate_factor=$(echo "print (0.5/$xent_regularize)" | python)
  opts="l2-regularize=0.0015 dropout-proportion=0.0 dropout-per-dim=true dropout-per-dim-continuous=true"
  linear_opts="l2-regularize=0.0015 orthonormal-constraint=-1.0"
  output_opts="l2-regularize=0.001"

  mkdir -p $dir/configs

  cat <<EOF > $dir/configs/network.xconfig
  input dim=100 name=ivector
  input dim=40 name=input

  # please note that it is important to have input layer with the name=input
  # as the layer immediately preceding the fixed-affine-layer to enable
  # the use of short notation for the descriptor
  fixed-affine-layer name=lda input=Append(-1,0,1,ReplaceIndex(ivector, t, 0)) affine-transform-file=$dir/configs/lda.mat

  # the first splicing is moved before the lda layer, so no splicing here
  relu-batchnorm-dropout-layer name=tdnn1 $opts dim=1536
  linear-component name=tdnn2l0 dim=320 $linear_opts input=Append(-1,0)
  linear-component name=tdnn2l dim=320 $linear_opts input=Append(-1,0)
  relu-batchnorm-dropout-layer name=tdnn2 $opts input=Append(0,1) dim=1536
  linear-component name=tdnn3l dim=320 $linear_opts input=Append(-1,0)
  relu-batchnorm-dropout-layer name=tdnn3 $opts dim=1536 input=Append(0,1)
  linear-component name=tdnn4l0 dim=320 $linear_opts input=Append(-1,0)
  linear-component name=tdnn4l dim=320 $linear_opts input=Append(0,1)
  relu-batchnorm-dropout-layer name=tdnn4 $opts input=Append(0,1) dim=1536
  linear-component name=tdnn5l dim=320 $linear_opts
  relu-batchnorm-dropout-layer name=tdnn5 $opts dim=1536 input=Append(0, tdnn3l)
  linear-component name=tdnn6l0 dim=320 $linear_opts input=Append(-3,0)
  linear-component name=tdnn6l dim=320 $linear_opts input=Append(-3,0)
  relu-batchnorm-dropout-layer name=tdnn6 $opts input=Append(0,3) dim=1792
  linear-component name=tdnn7l0 dim=320 $linear_opts input=Append(-3,0)
  linear-component name=tdnn7l dim=320 $linear_opts input=Append(0,3)
  relu-batchnorm-dropout-layer name=tdnn7 $opts input=Append(0,3,tdnn6l,tdnn4l,tdnn2l) dim=1536
  linear-component name=tdnn8l0 dim=320 $linear_opts input=Append(-3,0)
  linear-component name=tdnn8l dim=320 $linear_opts input=Append(0,3)
  relu-batchnorm-dropout-layer name=tdnn8 $opts input=Append(0,3) dim=1792
  linear-component name=tdnn9l0 dim=320 $linear_opts input=Append(-3,0)
  linear-component name=tdnn9l dim=320 $linear_opts input=Append(-3,0)
  relu-batchnorm-dropout-layer name=tdnn9 $opts input=Append(0,3,tdnn8l,tdnn6l,tdnn5l) dim=1536
  linear-component name=tdnn10l0 dim=320 $linear_opts input=Append(-3,0)
  linear-component name=tdnn10l dim=320 $linear_opts input=Append(0,3)
  relu-batchnorm-dropout-layer name=tdnn10 $opts input=Append(0,3) dim=1792
  linear-component name=tdnn11l0 dim=320 $linear_opts input=Append(-3,0)
  linear-component name=tdnn11l dim=320 $linear_opts input=Append(-3,0)
  relu-batchnorm-dropout-layer name=tdnn11 $opts input=Append(0,3,tdnn10l,tdnn9l,tdnn7l) dim=1536
  linear-component name=prefinal-l dim=320 $linear_opts

  relu-batchnorm-layer name=prefinal-chain input=prefinal-l $opts dim=1792
  linear-component name=prefinal-chain-l dim=320 $linear_opts
  batchnorm-component name=prefinal-chain-batchnorm
  output-layer name=output include-log-softmax=false dim=$num_targets $output_opts

  relu-batchnorm-layer name=prefinal-xent input=prefinal-l $opts dim=1792
  linear-component name=prefinal-xent-l dim=320 $linear_opts
  batchnorm-component name=prefinal-xent-batchnorm
  output-layer name=output-xent dim=$num_targets learning-rate-factor=$learning_rate_factor $output_opts
EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
fi

if [ $stage -le 13 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b{03,07,09,18}/$USER/kaldi-data/egs/multi-en-$(date +'%m_%d_%H_%M')/s5c/$dir/egs/storage $dir/egs/storage
  fi

  steps/nnet3/chain/train.py --stage $train_stage \
    --cmd "$decode_cmd" \
    --feat.online-ivector-dir exp/multi_a/nnet3/ivectors_${train_set} \
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
    --egs.opts "--frames-overlap-per-eg 0" \
    --egs.chunk-width $frames_per_eg \
    --trainer.num-chunk-per-minibatch $minibatch_size \
    --trainer.frames-per-iter 1500000 \
    --trainer.num-epochs $num_epochs \
    --trainer.dropout-schedule $dropout_schedule \
    --trainer.add-option="--optimization.memory-compression-level=2" \
    --trainer.optimization.num-jobs-initial $num_jobs_initial \
    --trainer.optimization.num-jobs-final $num_jobs_final \
    --trainer.optimization.initial-effective-lrate $initial_effective_lrate \
    --trainer.optimization.final-effective-lrate $final_effective_lrate \
    --trainer.max-param-change $max_param_change \
    --cleanup.remove-egs $remove_egs \
    --feat-dir data/${train_set}_hires \
    --tree-dir $treedir \
    --lat-dir exp/multi_a/tri5a_lats_nodup$suffix \
    --dir $dir  || exit 1;
fi

if [ $stage -le 14 ]; then
  # Note: it might appear that this $lang directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.
  utils/mkgraph.sh --self-loop-scale 1.0 data/lang_${multi}_${gmm}_fsh_sw1_tg $dir $dir/graph_fsh_sw1_tg
fi

decode_suff=fsh_sw1_tg
graph_dir=$dir/graph_fsh_sw1_tg

if [ $stage -le 15 ]; then
  if [ ! -z $decode_iter ]; then
    iter_opts=" --iter $decode_iter "
  fi
  if $rescore && [ ! -f data/lang_${multi}_${gmm}_fsh_sw1_fg/G.carpa ]; then
    LM_fg=data/local/lm/4gram-mincount/lm_unpruned.gz
    utils/build_const_arpa_lm.sh $LM_fg data/lang_${multi}_${gmm}_fsh_sw1_tg data/lang_${multi}_${gmm}_fsh_sw1_fg
  fi
  for decode_set in rt03 eval2000; do
      (
      steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
        --nj 50 --cmd "$decode_cmd" $iter_opts \
        --online-ivector-dir exp/multi_a/nnet3/ivectors_${decode_set} \
        $graph_dir data/${decode_set}_hires \
        $dir/decode_${decode_set}${decode_dir_affix:+_$decode_dir_affix}_${decode_suff} || exit 1;
      if $rescore; then
        steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
          data/lang_${multi}_${gmm}_fsh_sw1_{tg,fg} data/${decode_set}_hires \
          $dir/decode_${decode_set}${decode_dir_affix:+_$decode_dir_affix}_fsh_sw1_{tg,fg} || exit 1;
      fi
      ) &
  done
fi

test_online_decoding=true
lang=data/lang_${multi}_${gmm}_fsh_sw1_tg
if $test_online_decoding && [ $stage -le 16 ]; then
  # note: if the features change (e.g. you add pitch features), you will have to
  # change the options of the following command line.
  steps/online/nnet3/prepare_online_decoding.sh \
       --mfcc-config conf/mfcc_hires.conf \
       $lang exp/multi_a/nnet3/extractor $dir ${dir}_online

  rm $dir/.error 2>/dev/null || true
  for decode_set in rt03 eval2000; do
    (
      # note: we just give it "$decode_set" as it only uses the wav.scp, the
      # feature type does not matter.

      steps/online/nnet3/decode.sh --nj 50 --cmd "$decode_cmd" $iter_opts \
          --acwt 1.0 --post-decode-acwt 10.0 \
         $graph_dir data/${decode_set}_hires \
         ${dir}_online/decode_${decode_set}${decode_iter:+_$decode_iter}_${decode_suff} || exit 1;
      if $rescore; then
        steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
          data/lang_${multi}_${gmm}_fsh_sw1_{tg,fg} data/${decode_set}_hires \
          ${dir}_online/decode_${decode_set}${decode_dir_affix:+_$decode_dir_affix}_fsh_sw1_{tg,fg} || exit 1;
      fi
    ) || touch $dir/.error &
  done
  wait
  if [ -f $dir/.error ]; then
    echo "$0: something went wrong in online decoding"
    exit 1
  fi
fi

exit 0;
