#!/usr/bin/env bash
# Copyright 2018 Xiaohui Zhang
#           2017 University of Chinese Academy of Sciences (UCAS) Gaofeng Cheng
# Apache 2.0

# This is based on TDNN_LSTM_1b (from egs/swbd/s5c), but using the NormOPGRU to replace the LSTMP,
# and adding chunk-{left,right}-context-initial=0
# Different from the vanilla OPGRU, Norm-OPGRU adds batchnorm in its output (forward direction)
# and renorm in its recurrence. Experiments show that the TDNN-NormOPGRU could achieve similar
# results than TDNN-LSTMP and BLSTMP in both large or small data sets (80 ~ 2300 Hrs).

# ./local/chain/compare_wer_general.sh tdnn_5b_sp tdnn_opgru_1a_sp
# System                tdnn_5b_sp tdnn_opgru_1a_sp
# WER on eval2000(tg)        11.7      11.6
# WER on eval2000(fg)        11.5      11.5
# WER on rt03(tg)            11.9      11.5
# WER on rt03(fg)            11.5      11.2
# Final train prob          -0.097    -0.088
# Final valid prob          -0.090    -0.088
# Final train prob (xent)   -1.042    -1.048
# Final valid prob (xent)   -0.9712   -1.0253
# Num-parameters           34818416   37364848

# ./steps/info/chain_dir_info.pl exp/multi_a/chain/tdnn_opgru_1a_sp
# exp/multi_a/chain/tdnn_opgru_1a_sp: num-iters=2621 nj=3..16 num-params=37.4M dim=40+100->8504 combine=-0.082->-0.082 (over 2) 
# xent:train/valid[1744,2620,final]=(-1.62,-1.05,-1.05/-1.56,-1.02,-1.03) 
# logprob:train/valid[1744,2620,final]=(-0.118,-0.089,-0.088/-0.112,-0.089,-0.088)

# online results
# Eval2000
# %WER 14.5 | 2628 21594 | 87.6 8.9 3.6 2.1 14.5 49.3 | exp/multi_a/chain/tdnn_opgru_1a_sp_online/decode_eval2000_fsh_sw1_tg/score_8_0.0/eval2000_hires.ctm.callhm.filt.sys
# %WER 11.5 | 4459 42989 | 90.1 7.2 2.7 1.6 11.5 46.4 | exp/multi_a/chain/tdnn_opgru_1a_sp_online/decode_eval2000_fsh_sw1_tg/score_8_1.0/eval2000_hires.ctm.filt.sys
# %WER 8.4 | 1831 21395 | 92.8 5.3 1.9 1.1 8.4 41.8 | exp/multi_a/chain/tdnn_opgru_1a_sp_online/decode_eval2000_fsh_sw1_tg/score_10_0.0/eval2000_hires.ctm.swbd.filt.sys
# %WER 14.4 | 2628 21594 | 87.7 8.8 3.5 2.1 14.4 49.4 | exp/multi_a/chain/tdnn_opgru_1a_sp_online/decode_eval2000_fsh_sw1_fg/score_8_0.0/eval2000_hires.ctm.callhm.filt.sys
# %WER 11.4 | 4459 42989 | 90.2 7.1 2.7 1.7 11.4 46.3 | exp/multi_a/chain/tdnn_opgru_1a_sp_online/decode_eval2000_fsh_sw1_fg/score_8_1.0/eval2000_hires.ctm.filt.sys
# %WER 8.3 | 1831 21395 | 92.9 5.2 1.9 1.2 8.3 41.1 | exp/multi_a/chain/tdnn_opgru_1a_sp_online/decode_eval2000_fsh_sw1_fg/score_10_0.0/eval2000_hires.ctm.swbd.filt.sys

# RT03
# %WER 9.3 | 3970 36721 | 91.6 5.3 3.1 0.9 9.3 40.0 | exp/multi_a/chain/tdnn_opgru_1a_sp_online/decode_rt03_fsh_sw1_tg/score_8_0.0/rt03_hires.ctm.fsh.filt.sys
# %WER 11.4 | 8420 76157 | 89.8 6.7 3.5 1.2 11.4 42.1 | exp/multi_a/chain/tdnn_opgru_1a_sp_online/decode_rt03_fsh_sw1_tg/score_8_0.0/rt03_hires.ctm.filt.sys
# %WER 13.3 | 4450 39436 | 88.1 7.9 4.0 1.4 13.3 43.9 | exp/multi_a/chain/tdnn_opgru_1a_sp_online/decode_rt03_fsh_sw1_tg/score_8_0.5/rt03_hires.ctm.swbd.filt.sys
# %WER 9.2 | 3970 36721 | 91.9 5.4 2.7 1.1 9.2 39.6 | exp/multi_a/chain/tdnn_opgru_1a_sp_online/decode_rt03_fsh_sw1_fg/score_7_0.0/rt03_hires.ctm.fsh.filt.sys
# %WER 11.2 | 8420 76157 | 90.0 6.5 3.5 1.2 11.2 41.9 | exp/multi_a/chain/tdnn_opgru_1a_sp_online/decode_rt03_fsh_sw1_fg/score_8_0.0/rt03_hires.ctm.filt.sys
# %WER 13.1 | 4450 39436 | 88.3 7.8 3.9 1.4 13.1 43.6 | exp/multi_a/chain/tdnn_opgru_1a_sp_online/decode_rt03_fsh_sw1_fg/score_8_0.0/rt03_hires.ctm.swbd.filt.sys



set -e

# configs for 'chain'
stage=1
train_stage=576
get_egs_stage=-10
speed_perturb=true
multi=multi_a
gmm=tri5a
dir=exp/multi_a/chain/tdnn_opgru_1a # Note: _sp will get added to this if $speed_perturb == true.
decode_iter=
decode_dir_affix=
rescore=true # whether to rescore lattices
dropout_schedule='0,0@0.20,0.2@0.50,0'

# training options
leftmost_questions_truncate=-1
num_epochs=4
chunk_width=150
chunk_left_context=40
chunk_right_context=0
xent_regularize=0.025
self_repair_scale=0.00001
label_delay=5
# decode options
extra_left_context=50
extra_right_context=0
frames_per_chunk=

remove_egs=false
common_egs_dir=

affix=
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
  gru_opts="dropout-per-frame=true dropout-proportion=0.0 "

  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig
  input dim=100 name=ivector
  input dim=40 name=input

  # please note that it is important to have input layer with the name=input
  # as the layer immediately preceding the fixed-affine-layer to enable
  # the use of short notation for the descriptor
  fixed-affine-layer name=lda input=Append(-2,-1,0,1,2, ReplaceIndex(ivector, t, 0)) affine-transform-file=$dir/configs/lda.mat

  # the first splicing is moved before the lda layer, so no splicing here
  relu-batchnorm-layer name=tdnn1 dim=1024
  relu-batchnorm-layer name=tdnn2 input=Append(-1,0,1) dim=1024
  relu-batchnorm-layer name=tdnn3 input=Append(-1,0,1) dim=1024

  # check steps/libs/nnet3/xconfig/lstm.py for the other options and defaults
  norm-opgru-layer name=opgru1 cell-dim=1024 recurrent-projection-dim=256 non-recurrent-projection-dim=256 delay=-3 $gru_opts
  relu-batchnorm-layer name=tdnn4 input=Append(-3,0,3) dim=1024
  relu-batchnorm-layer name=tdnn5 input=Append(-3,0,3) dim=1024
  norm-opgru-layer name=opgru2 cell-dim=1024 recurrent-projection-dim=256 non-recurrent-projection-dim=256 delay=-3 $gru_opts
  relu-batchnorm-layer name=tdnn6 input=Append(-3,0,3) dim=1024
  relu-batchnorm-layer name=tdnn7 input=Append(-3,0,3) dim=1024
  norm-opgru-layer name=opgru3 cell-dim=1024 recurrent-projection-dim=256 non-recurrent-projection-dim=256 delay=-3 $gru_opts

  ## adding the layers for chain branch
  output-layer name=output input=opgru3 output-delay=$label_delay include-log-softmax=false dim=$num_targets max-change=1.5

  # adding the layers for xent branch
  # This block prints the configs for a separate output that will be
  # trained with a cross-entropy objective in the 'chain' models... this
  # has the effect of regularizing the hidden parts of the model.  we use
  # 0.5 / args.xent_regularize as the learning rate factor- the factor of
  # 0.5 / args.xent_regularize is suitable as it means the xent
  # final-layer learns at a rate independent of the regularization
  # constant; and the 0.5 was tuned so as to make the relative progress
  # similar in the xent and regular final layers.
  output-layer name=output-xent input=opgru3 output-delay=$label_delay dim=$num_targets learning-rate-factor=$learning_rate_factor max-change=1.5

EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
fi

if [ $stage -le 13 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{3,7,9,8}/$USER/kaldi-data/egs/multi-en-$(date +'%m_%d_%H_%M')/s5c/$dir/egs/storage $dir/egs/storage
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
    --trainer.num-chunk-per-minibatch 64 \
    --trainer.frames-per-iter 1200000 \
    --trainer.max-param-change 2.0 \
    --trainer.num-epochs $num_epochs \
    --trainer.optimization.shrink-value 0.99 \
    --trainer.optimization.num-jobs-initial 3 \
    --trainer.optimization.num-jobs-final 16 \
    --trainer.optimization.initial-effective-lrate 0.001 \
    --trainer.optimization.final-effective-lrate 0.0001 \
    --trainer.dropout-schedule $dropout_schedule \
    --trainer.optimization.momentum 0.0 \
    --trainer.deriv-truncate-margin 8 \
    --egs.stage $get_egs_stage \
    --egs.opts "--frames-overlap-per-eg 0" \
    --egs.chunk-width $chunk_width \
    --egs.chunk-left-context $chunk_left_context \
    --egs.chunk-right-context $chunk_right_context \
    --egs.chunk-left-context-initial 0 \
    --egs.chunk-right-context-final 0 \
    --egs.dir "$common_egs_dir" \
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
  rm $dir/.error 2>/dev/null || true
  [ -z $extra_left_context ] && extra_left_context=$chunk_left_context;
  [ -z $extra_right_context ] && extra_right_context=$chunk_right_context;
  [ -z $frames_per_chunk ] && frames_per_chunk=$chunk_width;
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
        --extra-left-context $extra_left_context  \
        --extra-right-context $extra_right_context  \
        --extra-left-context-initial 0 \
        --extra-right-context-final 0 \
        --frames-per-chunk "$frames_per_chunk" \
        --online-ivector-dir exp/multi_a/nnet3/ivectors_${decode_set} \
        $graph_dir data/${decode_set}_hires \
        $dir/decode_${decode_set}${decode_dir_affix:+_$decode_dir_affix}_${decode_suff} || exit 1;
      if $rescore; then
        steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
          data/lang_${multi}_${gmm}_fsh_sw1_{tg,fg} data/${decode_set}_hires \
          $dir/decode_${decode_set}${decode_dir_affix:+_$decode_dir_affix}_fsh_sw1_{tg,fg} || exit 1;
      fi
      ) || touch $dir/.error &
  done
  wait
  if [ -f $dir/.error ]; then
    echo "$0: something went wrong in decoding"
    exit 1
  fi
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
