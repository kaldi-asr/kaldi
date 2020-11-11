#!/usr/bin/env bash
# Copyright 2017 University of Chinese Academy of Sciences (UCAS) Gaofeng Cheng
# Apache 2.0

# Similar to swbd\s5c\local\chain\tuning\run_tdnn_lstm_1e.sh
# Difference between tdnn_lstm_1a and tdnn_lstm_1b:
# chunk width        150  140,100,160
# xent_regularize    0.025  0.01
# minibatch          64  64,32
# frames-per-iter    1200000  1500000
# batchnorm in TDNN  No  Yes
# Dropout in LSTM    No  Yes

# ./local/chain/compare_wer_general.sh --looped tdnn_lstm_1a_sp tdnn_lstm_1b_sp
# System                tdnn_lstm_1a_sp tdnn_lstm_1b_sp
# num-params                 39.7M     39.7M
# WER on eval2000(tg)        12.3      12.3
#           [looped:]        12.2      12.3
# WER on eval2000(fg)        12.1      12.0
#           [looped:]        12.1      12.2
# WER on rt03(tg)            11.6      11.4
#           [looped:]        11.6      11.6
# WER on rt03(fg)            11.3      11.1
#           [looped:]        11.3      11.3
# Final train prob         -0.074    -0.087
# Final valid prob         -0.084    -0.088
# Final train prob (xent)        -0.882    -1.015
# Final valid prob (xent)       -0.9393   -0.9837

#./steps/info/chain_dir_info.pl exp/chain/tdnn_lstm_1b_sp
#exp/chain/tdnn_lstm_1b_sp: num-iters=1909 nj=3..16 num-params=39.7M dim=40+100->6149 combine=-0.087->-0.086 (over 5) 
#xent:train/valid[1270,1908,final]=(-1.37,-1.02,-1.01/-1.31,-1.00,-0.984) 
#logprob:train/valid[1270,1908,final]=(-0.108,-0.088,-0.087/-0.103,-0.091,-0.088)


# online results
# Eval2000
#%WER 15.9 | 2628 21594 | 86.0 8.6 5.4 1.9 15.9 53.5 | exp/chain/tdnn_lstm_1b_online/decode_eval2000_fsh_sw1_tg/score_7_0.0/eval2000_hires.ctm.callhm.filt.sys
#%WER 12.3 | 4459 42989 | 89.1 6.8 4.1 1.5 12.3 49.2 | exp/chain/tdnn_lstm_1b_online/decode_eval2000_fsh_sw1_tg/score_8_0.0/eval2000_hires.ctm.filt.sys
#%WER 8.6 | 1831 21395 | 92.5 5.2 2.3 1.1 8.6 42.6 | exp/chain/tdnn_lstm_1b_online/decode_eval2000_fsh_sw1_tg/score_8_1.0/eval2000_hires.ctm.swbd.filt.sys
#%WER 15.7 | 2628 21594 | 86.2 8.5 5.3 1.9 15.7 53.0 | exp/chain/tdnn_lstm_1b_online/decode_eval2000_fsh_sw1_fg/score_7_0.0/eval2000_hires.ctm.callhm.filt.sys
#%WER 12.1 | 4459 42989 | 89.3 6.6 4.0 1.5 12.1 48.4 | exp/chain/tdnn_lstm_1b_online/decode_eval2000_fsh_sw1_fg/score_8_0.0/eval2000_hires.ctm.filt.sys
#%WER 8.5 | 1831 21395 | 92.5 4.9 2.5 1.0 8.5 41.1 | exp/chain/tdnn_lstm_1b_online/decode_eval2000_fsh_sw1_fg/score_10_0.0/eval2000_hires.ctm.swbd.filt.sys

# online results
# RT03
#%WER 9.4 | 3970 36721 | 91.4 5.0 3.5 0.9 9.4 39.5 | exp/chain/tdnn_lstm_1b_online/decode_rt03_fsh_sw1_tg/score_8_0.0/rt03_hires.ctm.fsh.filt.sys
#%WER 11.6 | 8420 76157 | 89.5 6.4 4.1 1.1 11.6 42.0 | exp/chain/tdnn_lstm_1b_online/decode_rt03_fsh_sw1_tg/score_8_0.0/rt03_hires.ctm.filt.sys
#%WER 13.5 | 4450 39436 | 87.6 7.3 5.0 1.1 13.5 44.5 | exp/chain/tdnn_lstm_1b_online/decode_rt03_fsh_sw1_tg/score_9_0.0/rt03_hires.ctm.swbd.filt.sys
#%WER 9.2 | 3970 36721 | 91.6 4.9 3.5 0.9 9.2 39.3 | exp/chain/tdnn_lstm_1b_online/decode_rt03_fsh_sw1_fg/score_8_0.0/rt03_hires.ctm.fsh.filt.sys
#%WER 11.3 | 8420 76157 | 89.8 6.2 4.0 1.1 11.3 41.6 | exp/chain/tdnn_lstm_1b_online/decode_rt03_fsh_sw1_fg/score_8_0.0/rt03_hires.ctm.filt.sys
#%WER 13.2 | 4450 39436 | 88.0 7.4 4.6 1.2 13.2 43.6 | exp/chain/tdnn_lstm_1b_online/decode_rt03_fsh_sw1_fg/score_8_0.0/rt03_hires.ctm.swbd.filt.sys

set -e

# configs for 'chain'
stage=12
train_stage=-10
get_egs_stage=-10
speed_perturb=true
dir=exp/chain/tdnn_lstm_1b # Note: _sp will get added to this if $speed_perturb == true.
decode_iter=
decode_dir_affix=

# training options
leftmost_questions_truncate=-1
frames_per_chunk=140,100,160
chunk_left_context=40
chunk_right_context=0
xent_regularize=0.01
self_repair_scale=0.00001
label_delay=5
dropout_schedule='0,0@0.20,0.2@0.50,0'
# decode options
extra_left_context=50
extra_right_context=0
frames_per_chunk_primary=$(echo $frames_per_chunk | cut -d, -f1)

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
  learning_rate_factor=$(echo "print (0.5/$xent_regularize)" | python)
  lstm_opts="decay-time=20 dropout-proportion=0.0"

  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig
  input dim=100 name=ivector
  input dim=40 name=input

  # please note that it is important to have input layer with the name=input
  # as the layer immediately preceding the fixed-affine-layer to enable
  # the use of short notation for the descriptor
  fixed-affine-layer name=lda input=Append(-2,-1,0,1,2,ReplaceIndex(ivector, t, 0)) affine-transform-file=$dir/configs/lda.mat

  # the first splicing is moved before the lda layer, so no splicing here
  relu-batchnorm-layer name=tdnn1 dim=1024
  relu-batchnorm-layer name=tdnn2 input=Append(-1,0,1) dim=1024
  relu-batchnorm-layer name=tdnn3 input=Append(-1,0,1) dim=1024

  # check steps/libs/nnet3/xconfig/lstm.py for the other options and defaults
  fast-lstmp-layer name=fastlstm1 cell-dim=1024 recurrent-projection-dim=256 non-recurrent-projection-dim=256 delay=-3 $lstm_opts
  relu-batchnorm-layer name=tdnn4 input=Append(-3,0,3) dim=1024
  relu-batchnorm-layer name=tdnn5 input=Append(-3,0,3) dim=1024
  fast-lstmp-layer name=fastlstm2 cell-dim=1024 recurrent-projection-dim=256 non-recurrent-projection-dim=256 delay=-3 $lstm_opts
  relu-batchnorm-layer name=tdnn6 input=Append(-3,0,3) dim=1024
  relu-batchnorm-layer name=tdnn7 input=Append(-3,0,3) dim=1024
  fast-lstmp-layer name=fastlstm3 cell-dim=1024 recurrent-projection-dim=256 non-recurrent-projection-dim=256 delay=-3 $lstm_opts

  ## adding the layers for chain branch
  output-layer name=output input=fastlstm3 output-delay=$label_delay include-log-softmax=false dim=$num_targets max-change=1.5

  # adding the layers for xent branch
  # This block prints the configs for a separate output that will be
  # trained with a cross-entropy objective in the 'chain' models... this
  # has the effect of regularizing the hidden parts of the model.  we use
  # 0.5 / args.xent_regularize as the learning rate factor- the factor of
  # 0.5 / args.xent_regularize is suitable as it means the xent
  # final-layer learns at a rate independent of the regularization
  # constant; and the 0.5 was tuned so as to make the relative progress
  # similar in the xent and regular final layers.
  output-layer name=output-xent input=fastlstm3 output-delay=$label_delay dim=$num_targets learning-rate-factor=$learning_rate_factor max-change=1.5

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
    --chain.l2-regularize 0.00005 \
    --chain.apply-deriv-weights false \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --trainer.num-chunk-per-minibatch 64,32 \
    --trainer.frames-per-iter 1500000 \
    --trainer.max-param-change 2.0 \
    --trainer.num-epochs 4 \
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
    --egs.chunk-width $frames_per_chunk \
    --egs.chunk-left-context $chunk_left_context \
    --egs.chunk-right-context $chunk_right_context \
    --egs.chunk-left-context-initial 0 \
    --egs.chunk-right-context-final 0 \
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
  rm $dir/.error 2>/dev/null || true
  [ -z $extra_left_context ] && extra_left_context=$chunk_left_context;
  [ -z $extra_right_context ] && extra_right_context=$chunk_right_context;
  if [ ! -z $decode_iter ]; then
    iter_opts=" --iter $decode_iter "
  fi
  for decode_set in rt03 eval2000; do
      (
      steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
          --nj 50 --cmd "$decode_cmd" $iter_opts \
          --extra-left-context $extra_left_context  \
          --extra-right-context $extra_right_context  \
          --extra-left-context-initial 0 \
          --extra-right-context-final 0 \
          --frames-per-chunk  "$frames_per_chunk_primary" \
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

test_online_decoding=false
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
