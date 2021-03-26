#!/usr/bin/env bash
# Copyright 2018 Xiaohui Zhang
#           2017 University of Chinese Academy of Sciences (UCAS) Gaofeng Cheng
# Apache 2.0

# This is similar with tdnn_opgru_1a but with correct num_leaves (7k rather than 11k),
# aligments from lattices when building the tree, and better l2-regularization as opgru-1a
# from fisher-swbd.

# ./local/chain/compare_wer_general.sh tdnn_opgru_1a_sp tdnn_opgru_1b_sp
# System            tdnn_opgru_1a_sp  tdnn_opgru_1b_sp
# WER on eval2000(tg)        11.6      11.4
# WER on eval2000(fg)        11.5      11.2
# WER on rt03(tg)            11.5      11.1
# WER on rt03(fg)            11.2      10.8
# Final train prob         -0.088     -0.091
# Final valid prob         -0.088     -0.091
# Final train prob (xent)  -1.048     -0.990
# Final valid prob (xent)  -1.0253    -0.091
# Num-parameters          37364848    34976320


# ./steps/info/chain_dir_info.pl exp/${multi}/chain/tdnn_opgru_1b_sp
# exp/${multi}/chain/tdnn_opgru_1b_sp: num-iters=2621 nj=3..16 num-params=35.0M dim=40+100->6176 combine=-0.098->-0.096 (over 4)
# xent:train/valid[1744,2620,final]=(-1.49,-0.991,-0.990/-1.51,-1.01,-1.01) 
# logprob:train/valid[1744,2620,final]=(-0.118,-0.091,-0.091/-0.117,-0.093,-0.091)

# online results
# Eval2000
# %WER 14.3 | 2628 21594 | 87.8 8.9 3.3 2.1 14.3 49.8 | exp/${multi}/chain/tdnn_opgru_1b_sp_online/decode_eval2000_fsh_sw1_tg/score_7_0.0/eval2000_hires.ctm.callhm.filt.sys
# %WER 11.4 | 4459 42989 | 90.2 7.2 2.7 1.6 11.4 46.3 | exp/${multi}/chain/tdnn_opgru_1b_sp_online/decode_eval2000_fsh_sw1_tg/score_8_0.5/eval2000_hires.ctm.filt.sys
# %WER 8.4 | 1831 21395 | 92.7 5.3 2.0 1.1 8.4 41.8 | exp/${multi}/chain/tdnn_opgru_1b_sp_online/decode_eval2000_fsh_sw1_tg/score_10_0.0/eval2000_hires.ctm.swbd.filt.sys
# %WER 14.2 | 2628 21594 | 88.0 8.8 3.3 2.2 14.2 49.4 | exp/${multi}/chain/tdnn_opgru_1b_sp_online/decode_eval2000_fsh_sw1_fg/score_7_0.0/eval2000_hires.ctm.callhm.filt.sys
# %WER 11.4 | 4459 42989 | 90.3 7.1 2.6 1.7 11.4 45.9 | exp/${multi}/chain/tdnn_opgru_1b_sp_online/decode_eval2000_fsh_sw1_fg/score_8_0.0/eval2000_hires.ctm.filt.sys
# %WER 8.2 | 1831 21395 | 92.9 5.1 2.0 1.1 8.2 41.3 | exp/${multi}/chain/tdnn_opgru_1b_sp_online/decode_eval2000_fsh_sw1_fg/score_11_0.0/eval2000_hires.ctm.swbd.filt.sys

# RT03
# %WER 9.0 | 3970 36721 | 92.0 5.5 2.4 1.1 9.0 37.9 | exp/${multi}/chain/tdnn_opgru_1b_sp_online/decode_rt03_fsh_sw1_tg/score_7_0.0/rt03_hires.ctm.fsh.filt.sys
# %WER 11.2 | 8420 76157 | 90.0 6.8 3.2 1.2 11.2 41.1 | exp/${multi}/chain/tdnn_opgru_1b_sp_online/decode_rt03_fsh_sw1_tg/score_8_0.0/rt03_hires.ctm.filt.sys
# %WER 13.2 | 4450 39436 | 88.1 7.5 4.4 1.3 13.2 44.1 | exp/${multi}/chain/tdnn_opgru_1b_sp_online/decode_rt03_fsh_sw1_tg/score_10_0.0/rt03_hires.ctm.swbd.filt.sys
# %WER 8.7 | 3970 36721 | 92.3 5.1 2.6 1.0 8.7 37.8 | exp/${multi}/chain/tdnn_opgru_1b_sp_online/decode_rt03_fsh_sw1_fg/score_8_0.0/rt03_hires.ctm.fsh.filt.sys
# %WER 10.9 | 8420 76157 | 90.3 6.5 3.1 1.2 10.9 40.6 | exp/${multi}/chain/tdnn_opgru_1b_sp_online/decode_rt03_fsh_sw1_fg/score_8_0.0/rt03_hires.ctm.filt.sys
# %WER 12.9 | 4450 39436 | 88.5 7.9 3.6 1.4 12.9 43.1 | exp/${multi}/chain/tdnn_opgru_1b_sp_online/decode_rt03_fsh_sw1_fg/score_8_0.0/rt03_hires.ctm.swbd.filt.sys

set -e

# configs for 'chain'
stage=-1
train_stage=-10
get_egs_stage=-10
speed_perturb=true
multi=multi_a
gmm=tri5a
dir=exp/${multi}/chain/tdnn_opgru_1b # Note: _sp will get added to this if $speed_perturb == true.
decode_iter=
decode_dir_affix=
rescore=true # whether to rescore lattices
dropout_schedule='0,0@0.20,0.2@0.50,0'

# training options
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
train_set=${multi}/${gmm}_sp
build_tree_ali_dir=exp/${multi}/${gmm}_ali_sp
treedir=exp/${multi}/chain/${gmm}_tree
lang=data/${multi}/lang_${gmm}_chain

# if we are using the speed-perturbed data we need to generate
# alignments for it.
local/nnet3/run_ivector_common.sh --stage $stage \
  --multi ${multi} \
  --gmm $gmm \
  --speed-perturb $speed_perturb \
  --generate-alignments $speed_perturb || exit 1;

if [ $stage -le 9 ]; then
  # Get the alignments as lattices (gives the LF-MMI training more freedom).
  # use the same num-jobs as the alignments
  nj=$(cat $build_tree_ali_dir/num_jobs) || exit 1;
  steps/align_fmllr_lats.sh --nj $nj --cmd "$train_cmd" \
    --generate-ali-from-lats true data/$train_set  \
    data/lang_${multi}_${gmm} exp/${multi}/$gmm exp/${multi}/${gmm}_lats_nodup$suffix
  rm exp/${multi}/${gmm}_lats_nodup$suffix/fsts.*.gz # save space
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
      --context-opts "--context-width=2 --central-position=1" \
      --cmd "$train_cmd" 7000 data/$train_set $lang exp/${multi}/${gmm}_lats_nodup$suffix $treedir
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
     /export/b0{3,7,9,8}/$USER/kaldi-data/egs/multi-en-$(date +'%m_%d_%H_%M')/s5/$dir/egs/storage $dir/egs/storage
  fi

  steps/nnet3/chain/train.py --stage $train_stage \
    --cmd "$decode_cmd" \
    --feat.online-ivector-dir exp/${multi}/nnet3/ivectors_${train_set} \
    --feat.cmvn-opts "--norm-means=false --norm-vars=false" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient 0.1 \
    --chain.l2-regularize 0.00005 \
    --chain.apply-deriv-weights false \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --trainer.num-chunk-per-minibatch 64 \
    --trainer.frames-per-iter 1200000 \
    --trainer.max-param-change 2.0 \
    --trainer.num-epochs $num_epocs \
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
    --lat-dir exp/${multi}/tri5a_lats_nodup$suffix \
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
        --online-ivector-dir exp/${multi}/nnet3/ivectors_${decode_set} \
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
       $lang exp/${multi}/nnet3/extractor $dir ${dir}_online

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
