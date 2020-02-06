#!/usr/bin/env bash
# Copyright 2017 University of Chinese Academy of Sciences (UCAS) Gaofeng Cheng
#           2018 Xiaohui Zhang
#           2018 Vimal Manohar
# Apache 2.0

# This recipe is similar with tdnn_lstm_1b recipefrom fisher_swbd/s5, and is currently
# the best performing multi-en recipe.

# System             tdnn_opgru_1b_sp  tdnn_lstm_1a_sp
# WER on eval2000(tg)           11.4     11.4
# WER on eval2000(fg)           11.2     11.2
# WER on rt03(tg)               11.1     10.7
# WER on rt03(fg)               10.8     10.5
# Final train prob            -0.091    -0.095
# Final valid prob             -0.091   -0.089
# Final train prob (xent)     -0.990    -0.970
# Final valid prob (xent)     -0.091    -0.9638
# Num-parameters            34976320    39704128

# ./steps/info/chain_dir_info.pl exp/multi_a/chain/tdnn_lstm_1a_sp
# exp/multi_a/chain/tdnn_lstm_1a_sp: num-iters=2096 nj=3..16 num-params=39.7M dim=40+100->6176 combine=-0.088->-0.087 (over 3) 
# xent:train/valid[1395,2095,final]=(-1.38,-0.960,-0.970/-1.39,-0.964,-0.964) 
# logprob:train/valid[1395,2095,final]=(-0.117,-0.091,-0.095/-0.109,-0.087,-0.089)

# online results
# Eval2000
# %WER 14.2 | 2628 21594 | 87.8 8.6 3.5 2.1 14.2 49.1 | exp/multi_a/chain/tdnn_lstm_1a_sp_online/decode_eval2000/score_8_0.0/eval2000_hires.ctm.callhm.filt.sys
# %WER 11.4 | 4459 42989 | 90.3 7.0 2.7 1.7 11.4 46.1 | exp/multi_a/chain/tdnn_lstm_1a_sp_online/decode_eval2000/score_8_0.0/eval2000_hires.ctm.filt.sys
# %WER 8.4 | 1831 21395 | 92.8 5.3 2.0 1.2 8.4 41.2 | exp/multi_a/chain/tdnn_lstm_1a_sp_online/decode_eval2000/score_9_0.0/eval2000_hires.ctm.swbd.filt.sys
# %WER 14.0 | 2628 21594 | 88.0 8.5 3.4 2.1 14.0 48.6 | exp/multi_a/chain/tdnn_lstm_1a_sp_online/decode_eval2000_fg/score_8_0.0/eval2000_hires.ctm.callhm.filt.sys
# %WER 11.2 | 4459 42989 | 90.5 6.9 2.6 1.7 11.2 45.4 | exp/multi_a/chain/tdnn_lstm_1a_sp_online/decode_eval2000_fg/score_8_0.0/eval2000_hires.ctm.filt.sys
# %WER 8.1 | 1831 21395 | 93.1 5.1 1.8 1.2 8.1 40.6 | exp/multi_a/chain/tdnn_lstm_1a_sp_online/decode_eval2000_fg/score_9_0.0/eval2000_hires.ctm.swbd.filt.sys

# RT03
# %WER 8.7 | 3970 36721 | 92.2 5.3 2.5 1.0 8.7 37.3 | exp/multi_a/chain/tdnn_lstm_1a_sp_online/decode_rt03/score_7_0.0/rt03_hires.ctm.fsh.filt.sys
# %WER 10.8 | 8420 76157 | 90.4 6.5 3.2 1.2 10.8 40.1 | exp/multi_a/chain/tdnn_lstm_1a_sp_online/decode_rt03/score_8_0.0/rt03_hires.ctm.filt.sys
# %WER 12.7 | 4450 39436 | 88.7 7.7 3.6 1.4 12.7 42.5 | exp/multi_a/chain/tdnn_lstm_1a_sp_online/decode_rt03/score_8_0.0/rt03_hires.ctm.swbd.filt.sys
# %WER 8.5 | 3970 36721 | 92.4 5.1 2.5 0.9 8.5 37.2 | exp/multi_a/chain/tdnn_lstm_1a_sp_online/decode_rt03_fg/score_7_1.0/rt03_hires.ctm.fsh.filt.sys
# %WER 10.5 | 8420 76157 | 90.6 6.3 3.1 1.2 10.5 40.1 | exp/multi_a/chain/tdnn_lstm_1a_sp_online/decode_rt03_fg/score_8_0.0/rt03_hires.ctm.filt.sys
# %WER 12.4 | 4450 39436 | 88.9 7.2 3.9 1.3 12.4 42.7 | exp/multi_a/chain/tdnn_lstm_1a_sp_online/decode_rt03_fg/score_9_0.0/rt03_hires.ctm.swbd.filt.sys

set -e

# configs for 'chain'
stage=-10
train_stage=-10
get_egs_stage=-10
speed_perturb=true
multi=multi_a
gmm=tri5a
decode_iter=
decode_dir_affix=
decode_nj=50

# training options
frames_per_chunk=140,100,160
frames_per_chunk_primary=$(echo $frames_per_chunk | cut -d, -f1)
chunk_left_context=40
chunk_right_context=0
xent_regularize=0.025
self_repair_scale=0.00001
label_delay=5
# decode options
extra_left_context=50
extra_right_context=0
dropout_schedule='0,0@0.20,0.3@0.50,0'
num_epochs=4

remove_egs=false
common_egs_dir=

test_online_decoding=true  # if true, it will run the last decoding stage.

nnet3_affix=
tdnn_affix=_1a

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

dir=exp/$multi/chain/tdnn_lstm${tdnn_affix}${suffix}
train_set=${multi}/${gmm}${suffix}
lats_dir=exp/${multi}/${gmm}_lats_nodup${suffix}
treedir=exp/$multi/chain/${gmm}_tree
lang=data/${multi}/lang_${gmm}_chain
lang_dir=data/lang_${multi}_${gmm}_fsh_sw1_tg
rescore_lang_dir=data/lang_${multi}_${gmm}_fsh_sw1_fg

local/nnet3/run_ivector_common.sh --stage $stage \
  --multi $multi \
  --gmm $gmm \
  --speed-perturb $speed_perturb || exit 1

online_ivector_dir=exp/$multi/nnet3${nnet3_affix}/ivectors_${train_set}

if [ $stage -le 9 ]; then
  steps/align_fmllr_lats.sh --nj 100 --cmd "$train_cmd" \
    --generate-ali-from-lats true data/$train_set \
    data/lang_${multi}_${gmm} exp/${multi}/$gmm $lats_dir
  rm ${lats_dir}/fsts.*.gz  # save space
fi

if [ $stage -le 10 ]; then
  # Create a version of the lang/ directory that has one state per phone in the
  # topo file. [note, it really has two states.. the first one is only repeated
  # once, the second one has zero or more repeats.]
  if [ -d $lang ]; then 
    echo "$lang exists. Remove it or skip this stage."
    exit 1
  fi

  cp -r data/lang_${multi}_${gmm} $lang
  silphonelist=$(cat $lang/phones/silence.csl) || exit 1;
  nonsilphonelist=$(cat $lang/phones/nonsilence.csl) || exit 1;
  # Use our special topology... note that later on may have to tune this
  # topology.
  steps/nnet3/chain/gen_topo.py $nonsilphonelist $silphonelist >$lang/topo
fi

if [ $stage -le 11 ]; then
  # Build a tree using our new topology.

  if [ -f $treedir/final.mdl ]; then 
    echo "$treedir exists. Remove it or skip this stage."
    exit 1
  fi

  steps/nnet3/chain/build_tree.sh --frame-subsampling-factor 3 \
      --context-opts "--context-width=2 --central-position=1" \
      --cmd "$train_cmd" 7000 data/$train_set $lang $lats_dir $treedir
fi

if [ $stage -le 12 ]; then
  echo "$0: creating neural net configs using the xconfig parser";

  num_targets=$(tree-info $treedir/tree |grep num-pdfs|awk '{print $2}')
  learning_rate_factor=$(echo "print (0.5/$xent_regularize)" | python)
  lstm_opts="dropout-proportion=0.0 decay-time=40"

  relu_dim=1024
  cell_dim=1024
  projection_dim=256

  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig
  input dim=100 name=ivector
  input dim=40 name=input

  # please note that it is important to have input layer with the name=input
  # as the layer immediately preceding the fixed-affine-layer to enable
  # the use of short notation for the descriptor
  fixed-affine-layer name=lda input=Append(-2,-1,0,1,2,ReplaceIndex(ivector, t, 0)) affine-transform-file=$dir/configs/lda.mat

  # the first splicing is moved before the lda layer, so no splicing here
  relu-batchnorm-layer name=tdnn1 dim=$relu_dim
  relu-batchnorm-layer name=tdnn2 input=Append(-1,0,1) dim=$relu_dim
  relu-batchnorm-layer name=tdnn3 input=Append(-1,0,1) dim=$relu_dim

  # check steps/libs/nnet3/xconfig/lstm.py for the other options and defaults
  fast-lstmp-layer name=lstm1 cell-dim=$cell_dim recurrent-projection-dim=$projection_dim non-recurrent-projection-dim=$projection_dim delay=-3 $lstm_opts
  relu-batchnorm-layer name=tdnn4 input=Append(-3,0,3) dim=$relu_dim
  relu-batchnorm-layer name=tdnn5 input=Append(-3,0,3) dim=$relu_dim
  fast-lstmp-layer name=lstm2 cell-dim=$cell_dim recurrent-projection-dim=$projection_dim non-recurrent-projection-dim=$projection_dim delay=-3 $lstm_opts
  relu-batchnorm-layer name=tdnn6 input=Append(-3,0,3) dim=$relu_dim
  relu-batchnorm-layer name=tdnn7 input=Append(-3,0,3) dim=$relu_dim
  fast-lstmp-layer name=lstm3 cell-dim=$cell_dim recurrent-projection-dim=$projection_dim non-recurrent-projection-dim=$projection_dim delay=-3 $lstm_opts

  ## adding the layers for chain branch
  output-layer name=output input=lstm3 output-delay=$label_delay include-log-softmax=false dim=$num_targets max-change=1.5

  # adding the layers for xent branch
  # This block prints the configs for a separate output that will be
  # trained with a cross-entropy objective in the 'chain' models... this
  # has the effect of regularizing the hidden parts of the model.  we use
  # 0.5 / args.xent_regularize as the learning rate factor- the factor of
  # 0.5 / args.xent_regularize is suitable as it means the xent
  # final-layer learns at a rate independent of the regularization
  # constant; and the 0.5 was tuned so as to make the relative progress
  # similar in the xent and regular final layers.
  output-layer name=output-xent input=lstm3 output-delay=$label_delay dim=$num_targets learning-rate-factor=$learning_rate_factor max-change=1.5

EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
fi

if [ $stage -le 13 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
     /export/b0{5,6,7,8}/$USER/kaldi-data/egs/multi-en-$(date +'%m_%d_%H_%M')/s5/$dir/egs/storage $dir/egs/storage
  fi

  steps/nnet3/chain/train.py --stage $train_stage \
    --cmd "$decode_cmd" \
    --feat.online-ivector-dir exp/$multi/nnet3${nnet3_affix}/ivectors_${train_set} \
    --feat.cmvn-opts "--norm-means=false --norm-vars=false" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient 0.1 \
    --chain.l2-regularize 0.00005 \
    --chain.apply-deriv-weights false \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --trainer.num-chunk-per-minibatch 64,32 \
    --trainer.frames-per-iter 1500000 \
    --trainer.max-param-change 2.0 \
    --trainer.num-epochs $num_epochs \
    --trainer.optimization.shrink-value 0.99 \
    --trainer.optimization.num-jobs-initial 3 \
    --trainer.optimization.num-jobs-final 16 \
    --trainer.optimization.initial-effective-lrate 0.001 \
    --trainer.optimization.final-effective-lrate 0.0001 \
    --trainer.dropout-schedule=$dropout_schedule \
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
    --lat-dir $lats_dir \
    --dir $dir  || exit 1;
fi

lang_suffix=${lang_dir##*lang}

if [ $stage -le 14 ]; then
  # Note: it might appear that this $lang directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.
  utils/mkgraph.sh --self-loop-scale 1.0 $lang_dir \
    $dir $dir/graph${lang_suffix}
fi

graph_dir=$dir/graph${lang_suffix}
if [ $stage -le 15 ]; then
  iter_opts=
  [ -z $extra_left_context ] && extra_left_context=$chunk_left_context;
  [ -z $extra_right_context ] && extra_right_context=$chunk_right_context;
  if [ ! -z $decode_iter ]; then
    iter_opts=" --iter $decode_iter "
  fi
  for decode_set in eval2000 rt03; do
      (
      steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
        --nj 50 --cmd "$decode_cmd" $iter_opts \
        --extra-left-context $extra_left_context \
        --extra-right-context $extra_right_context \
        --extra-left-context-initial 0 \
        --extra-right-context-final 0 \
        --frames-per-chunk "$frames_per_chunk_primary" \
        --online-ivector-dir exp/$multi/nnet3${nnet3_affix}/ivectors_${decode_set} \
        $graph_dir data/${decode_set}_hires \
         $dir/decode${lang_suffix}_${decode_set}${decode_dir_affix:+_$decode_dir_affix}${decode_iter:+_iter$decode_iter} 
      
      steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
          $lang_dir $rescore_lang_dir data/${decode_set}_hires \
          $dir/decode${lang_suffix}_${decode_set}${decode_dir_affix:+_$decode_dir_affix}{,_fg}${decode_iter:+_iter$decode_iter} || exit 1;
      ) &
  done
fi
wait;

if $test_online_decoding && [ $stage -le 16 ]; then
  # note: if the features change (e.g. you add pitch features), you will have to
  # change the options of the following command line.
  steps/online/nnet3/prepare_online_decoding.sh \
       --mfcc-config conf/mfcc_hires.conf \
       $lang exp/nnet3/extractor $dir ${dir}_online

  rm $dir/.error 2>/dev/null || true
  for decode_set in train_dev eval2000; do
    (
      # note: we just give it "$decode_set" as it only uses the wav.scp, the
      # feature type does not matter.

      steps/online/nnet3/decode.sh --nj $decode_nj --cmd "$decode_cmd" $iter_opts \
          --acwt 1.0 --post-decode-acwt 10.0 \
         $graph_dir data/${decode_set}_hires \
         ${dir}_online/decode_${decode_set}${decode_iter:+_$decode_iter}_sw1_tg || exit 1;
      if $has_fisher; then
          steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
            data/lang_sw1_{tg,fsh_fg} data/${decode_set}_hires \
            ${dir}_online/decode_${decode_set}${decode_iter:+_$decode_iter}_sw1_{tg,fsh_fg} || exit 1;
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
