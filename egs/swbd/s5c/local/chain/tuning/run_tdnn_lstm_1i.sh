#!/bin/bash

# run_tdnn_lstm_1i.sh is like run_tdnn_lstm_1{e,f}.sh but
# with a different frames-per-iter: 2 million, vs. 1.5 million
# (1e) and 1 million (1f)

# Results are inconclusive regarding comparison with 1e: it's [0.3 worse, 0.1
# better, 0.2 worse, same, 0.2 better, 0.2 better, 0.3 better, 0.3 better] on
# the different conditions.  There is less train/valid difference and worse
# train prob [the trends of valid and train probs are consistent as we change
# the frames-per-iter].

# local/chain/compare_wer_general.sh --looped tdnn_lstm_1{e,f,i}_sp 2>/dev/null
# System                tdnn_lstm_1e_sp tdnn_lstm_1f_sp tdnn_lstm_1i_sp
# WER on train_dev(tg)      12.74     13.23     13.08
#           [looped:]       12.93     13.27     13.00
# WER on train_dev(fg)      11.70     12.17     11.97
#           [looped:]       12.09     12.42     12.08
# WER on eval2000(tg)        15.7      16.1      15.5
#           [looped:]        15.9      16.2      15.7
# WER on eval2000(fg)        14.3      14.6      14.0
#           [looped:]        14.6      14.7      14.3
# Final train prob         -0.066    -0.065    -0.069
# Final valid prob         -0.087    -0.090    -0.088
# Final train prob (xent)        -0.931    -0.916    -0.947
# Final valid prob (xent)       -1.0279   -1.0359   -1.0419

# run_tdnn_lstm_1e.sh is like run_tdnn_lstm_1d.sh but
# trying the change of xent_regularize from 0.025 (which was an
# unusual value) to the more usual 0.01.

# WER is worse but this seems to be due to more complete optimization
# (train better, valid worse).  Looks like we may be overtraining.
#
# local/chain/compare_wer_general.sh --looped tdnn_lstm_1e_sp tdnn_lstm_1f_sp
# System                tdnn_lstm_1e_sp tdnn_lstm_1f_sp
# WER on train_dev(tg)      12.74     13.23
#           [looped:]       12.93     13.27
# WER on train_dev(fg)      11.70     12.17
#           [looped:]       12.09     12.42
# WER on eval2000(tg)        15.7      16.1
#           [looped:]        15.9      16.2
# WER on eval2000(fg)        14.3      14.6
#           [looped:]        14.6      14.7
# Final train prob         -0.066    -0.065
# Final valid prob         -0.087    -0.090
# Final train prob (xent)        -0.931    -0.916
# Final valid prob (xent)       -1.0279   -1.0359


set -e

# configs for 'chain'
stage=0
train_stage=-10
get_egs_stage=-10
speed_perturb=true
dir=exp/chain/tdnn_lstm_1i # Note: _sp will get added to this if $speed_perturb == true.
decode_iter=final

# training options
xent_regularize=0.01
label_delay=5

chunk_left_context=40
chunk_right_context=0
# we'll put chunk-left-context-initial=0 and chunk-right-context-final=0
# directly without variables.
frames_per_chunk=140,100,160

# (non-looped) decoding options
frames_per_chunk_primary=$(echo $frames_per_chunk | cut -d, -f1)
extra_left_context=50
extra_right_context=0
# we'll put extra-left-context-initial=0 and extra-right-context-final=0
# directly without variables.


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

# The iVector-extraction and feature-dumping parts are the same as the standard
# nnet3 setup, and you can skip them by setting "--stage 8" if you have already
# run those things.

suffix=
if [ "$speed_perturb" == "true" ]; then
  suffix=_sp
fi

dir=${dir}$suffix
train_set=train_nodup$suffix
ali_dir=exp/tri4_ali_nodup$suffix
treedir=exp/chain/tri5_7d_tree$suffix
lang=data/lang_chain_2y


# if we are using the speed-perturbed data we need to generate
# alignments for it.
local/nnet3/run_ivector_common.sh --stage $stage \
  --speed-perturb $speed_perturb \
  --generate-alignments $speed_perturb || exit 1;


if [ $stage -le 9 ]; then
  # Get the alignments as lattices (gives the CTC training more freedom).
  # use the same num-jobs as the alignments
  nj=$(cat exp/tri4_ali_nodup$suffix/num_jobs) || exit 1;
  steps/align_fmllr_lats.sh --nj $nj --cmd "$train_cmd" data/$train_set \
    data/lang exp/tri4 exp/tri4_lats_nodup$suffix
  rm exp/tri4_lats_nodup$suffix/fsts.*.gz # save space
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
      --context-opts "--context-width=2 --central-position=1" \
      --cmd "$train_cmd" 7000 data/$train_set $lang $ali_dir $treedir
fi

if [ $stage -le 12 ]; then
  echo "$0: creating neural net configs using the xconfig parser";

  num_targets=$(tree-info $treedir/tree |grep num-pdfs|awk '{print $2}')
  [ -z $num_targets ] && { echo "$0: error getting num-targets"; exit 1; }
  learning_rate_factor=$(echo "print 0.5/$xent_regularize" | python)

  lstm_opts="decay-time=20"

  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig
  input dim=100 name=ivector
  input dim=40 name=input

  # please note that it is important to have input layer with the name=input
  # as the layer immediately preceding the fixed-affine-layer to enable
  # the use of short notation for the descriptor
  fixed-affine-layer name=lda input=Append(-2,-1,0,1,2,ReplaceIndex(ivector, t, 0)) affine-transform-file=$dir/configs/lda.mat

  # the first splicing is moved before the lda layer, so no splicing here
  relu-renorm-layer name=tdnn1 dim=1024
  relu-renorm-layer name=tdnn2 input=Append(-1,0,1) dim=1024
  relu-renorm-layer name=tdnn3 input=Append(-1,0,1) dim=1024

  # check steps/libs/nnet3/xconfig/lstm.py for the other options and defaults
  fast-lstmp-layer name=fastlstm1 cell-dim=1024 recurrent-projection-dim=256 non-recurrent-projection-dim=256 delay=-3 $lstm_opts
  relu-renorm-layer name=tdnn4 input=Append(-3,0,3) dim=1024
  relu-renorm-layer name=tdnn5 input=Append(-3,0,3) dim=1024
  fast-lstmp-layer name=fastlstm2 cell-dim=1024 recurrent-projection-dim=256 non-recurrent-projection-dim=256 delay=-3 $lstm_opts
  relu-renorm-layer name=tdnn6 input=Append(-3,0,3) dim=1024
  relu-renorm-layer name=tdnn7 input=Append(-3,0,3) dim=1024
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
     /export/b1{1,2,3,4}/$USER/kaldi-data/egs/swbd-$(date +'%m_%d_%H_%M')/s5c/$dir/egs/storage $dir/egs/storage
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
    --trainer.frames-per-iter 2000000 \
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
    --egs.chunk-width $frames_per_chunk \
    --egs.chunk-left-context $chunk_left_context \
    --egs.chunk-right-context $chunk_right_context \
    --egs.chunk-left-context-initial 0 \
    --egs.chunk-right-context-final 0 \
    --egs.dir "$common_egs_dir" \
    --cleanup.remove-egs $remove_egs \
    --feat-dir data/${train_set}_hires \
    --tree-dir $treedir \
    --lat-dir exp/tri4_lats_nodup$suffix \
    --dir $dir  || exit 1;
fi

if [ $stage -le 14 ]; then
  # Note: it might appear that this $lang directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.
  utils/mkgraph.sh --self-loop-scale 1.0 data/lang_sw1_tg $dir $dir/graph_sw1_tg
fi


graph_dir=$dir/graph_sw1_tg
if [ $stage -le 15 ]; then
  for decode_set in train_dev eval2000; do
      (
        steps/nnet3/decode.sh --num-threads 4 \
          --acwt 1.0 --post-decode-acwt 10.0 \
          --nj 25 --cmd "$decode_cmd" --iter $decode_iter \
          --extra-left-context $extra_left_context  \
          --extra-right-context $extra_right_context  \
          --extra-left-context-initial 0 \
          --extra-right-context-final 0 \
          --frames-per-chunk "$frames_per_chunk_primary" \
          --online-ivector-dir exp/nnet3/ivectors_${decode_set} \
         $graph_dir data/${decode_set}_hires \
         $dir/decode_${decode_set}_sw1_tg || exit 1;
      if $has_fisher; then
          steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
            data/lang_sw1_{tg,fsh_fg} data/${decode_set}_hires \
            $dir/decode_${decode_set}_sw1_{tg,fsh_fg} || exit 1;
      fi
      ) &
  done
fi
wait;

if [ $stage -le 16 ]; then
  # looped decoding.  Note: this does not make sense for BLSTMs or other
  # backward-recurrent setups, and for TDNNs and other non-recurrent there is no
  # point doing it because it would give identical results to regular decoding.
  for decode_set in train_dev eval2000; do
    (
      steps/nnet3/decode_looped.sh \
         --acwt 1.0 --post-decode-acwt 10.0 \
         --nj 50 --cmd "$decode_cmd" --iter $decode_iter \
         --online-ivector-dir exp/nnet3/ivectors_${decode_set} \
         $graph_dir data/${decode_set}_hires \
         $dir/decode_${decode_set}_sw1_tg_looped || exit 1;
      if $has_fisher; then
          steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
            data/lang_sw1_{tg,fsh_fg} data/${decode_set}_hires \
            $dir/decode_${decode_set}_sw1_{tg,fsh_fg}_looped || exit 1;
      fi
      ) &
  done
fi
wait;



exit 0;
