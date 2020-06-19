#!/usr/bin/env bash
# this is the tdnn-lstmp based on the run_tdnn_lstm_1a.sh under Librispeech but with larger model size.

# training acoustic model and decoding:
#     local/chain/tuning/run_tdnn_lstm_1b.sh
# local/chain/compare_wer.sh exp/chain_cleaned/tdnn_lstm1a_sp exp/chain_cleaned/tdnn_lstm1b_sp
# System                      tdnn_lstm1a_sp tdnn_lstm1b_sp
# WER on dev(fglarge)              3.44      3.36
# WER on dev(tglarge)              3.55      3.48
# WER on dev(tgmed)                4.41      4.26
# WER on dev(tgsmall)              4.82      4.71
# WER on dev_other(fglarge)        8.63      8.43
# WER on dev_other(tglarge)        9.09      8.94
# WER on dev_other(tgmed)         10.99     10.65
# WER on dev_other(tgsmall)       11.95     11.51
# WER on test(fglarge)             3.78      3.83
# WER on test(tglarge)             3.94      3.93
# WER on test(tgmed)               4.68      4.72
# WER on test(tgsmall)             5.11      5.10
# WER on test_other(fglarge)       8.83      8.69
# WER on test_other(tglarge)       9.09      9.10
# WER on test_other(tgmed)        11.05     10.86
# WER on test_other(tgsmall)      12.18     11.83
# Final train prob              -0.0452   -0.0417
# Final valid prob              -0.0477   -0.0459
# Final train prob (xent)       -0.7874   -0.7488
# Final valid prob (xent)       -0.8150   -0.7757
# Num-parameters               27790288  45245520

# rnn-lm rescoring:
#     local/rnnlm/tuning/run_tdnn_lstm_1a.sh --ac-model-dir exp/chain_cleaned/tdnn_lstm1b_sp/
# System                      tdnn_lstm1b_sp
# WER on dev(fglarge_nbe_rnnlm)      2.73
# WER on dev(fglarge_lat_rnnlm)        2.83
# WER on dev(fglarge)              3.36
# WER on dev(tglarge)              3.48
# WER on dev_other(fglarge_nbe_rnnlm)      7.20
# WER on dev_other(fglarge_lat_rnnlm)      7.23
# WER on dev_other(fglarge)        8.43
# WER on dev_other(tglarge)        8.94
# WER on test(fglarge_nbe_rnnlm)      3.10
# WER on test(fglarge_lat_rnnlm)       3.22
# WER on test(fglarge)             3.83
# WER on test(tglarge)             3.93
# WER on test_other(fglarge_nbe_rnnlm)      7.54
# WER on test_other(fglarge_lat_rnnlm)      7.65
# WER on test_other(fglarge)       8.69
# WER on test_other(tglarge)       9.10
# Final train prob              -0.0417
# Final valid prob              -0.0459
# Final train prob (xent)       -0.7488
# Final valid prob (xent)       -0.7757
# Num-parameters               45245520



set -e

# configs for 'chain'
stage=12
train_stage=-10
get_egs_stage=-10
speed_perturb=true
affix=1b
decode_iter=
decode_nj=50

# LSTM training options
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

remove_egs=false
common_egs_dir=
nnet3_affix=_cleaned
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

gmm=tri6b_cleaned
dir=exp/chain${nnet3_affix}/tdnn_lstm${affix}${suffix}
train_set=train_960_cleaned
ali_dir=exp/${gmm}_ali_${train_set}_sp_comb
tree_dir=exp/chain${nnet3_affix}/tree_sp${tree_affix:+_$tree_affix}
lang=data/lang_chain
train_data_dir=data/${train_set}_sp_hires_comb
lores_train_data_dir=data/${train_set}_sp_comb
train_ivector_dir=exp/nnet3${nnet3_affix}/ivectors_${train_set}_sp_hires_comb
lat_dir=exp/chain${nnet3_affix}/${gmm}_${train_set}_sp_comb_lats

if [ $stage -le 12 ]; then
  echo "$0: creating neural net configs using the xconfig parser";

  num_targets=$(tree-info $tree_dir/tree |grep num-pdfs|awk '{print $2}')
  learning_rate_factor=$(echo "print (0.5/$xent_regularize)" | python)

  opts="l2-regularize=0.002"
  linear_opts="orthonormal-constraint=1.0"
  lstm_opts="l2-regularize=0.0005 decay-time=40"
  output_opts="l2-regularize=0.0005 output-delay=$label_delay max-change=1.5 dim=$num_targets"


  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig
  input dim=100 name=ivector
  input dim=40 name=input

  fixed-affine-layer name=lda input=Append(-1,0,1,ReplaceIndex(ivector, t, 0)) affine-transform-file=$dir/configs/lda.mat

  # the first splicing is moved before the lda layer, so no splicing here
  relu-batchnorm-layer name=tdnn1 $opts dim=1280
  linear-component name=tdnn2l dim=320 $linear_opts input=Append(-1,0)
  relu-batchnorm-layer name=tdnn2 $opts input=Append(0,1) dim=1280
  linear-component name=tdnn3l dim=320 $linear_opts
  relu-batchnorm-layer name=tdnn3 $opts dim=1280
  linear-component name=tdnn4l dim=320 $linear_opts input=Append(-1,0)
  relu-batchnorm-layer name=tdnn4 $opts input=Append(0,1) dim=1280
  linear-component name=tdnn5l dim=320 $linear_opts
  relu-batchnorm-layer name=tdnn5 $opts dim=1280 input=Append(tdnn5l, tdnn3l)
  linear-component name=tdnn6l dim=320 $linear_opts input=Append(-3,0)
  relu-batchnorm-layer name=tdnn6 $opts input=Append(0,3) dim=1280
  linear-component name=lstm1l dim=320 $linear_opts input=Append(-3,0)
  fast-lstmp-layer name=lstm1 cell-dim=1536 recurrent-projection-dim=384 non-recurrent-projection-dim=384 delay=-3 dropout-proportion=0.0 $lstm_opts
  relu-batchnorm-layer name=tdnn7 $opts input=Append(0,3,tdnn6l,tdnn4l,tdnn2l) dim=1280
  linear-component name=tdnn8l dim=320 $linear_opts input=Append(-3,0)
  relu-batchnorm-layer name=tdnn8 $opts input=Append(0,3) dim=1280
  linear-component name=lstm2l dim=320 $linear_opts input=Append(-3,0)
  fast-lstmp-layer name=lstm2 cell-dim=1536 recurrent-projection-dim=384 non-recurrent-projection-dim=384 delay=-3 dropout-proportion=0.0 $lstm_opts
  relu-batchnorm-layer name=tdnn9 $opts input=Append(0,3,tdnn8l,tdnn6l,tdnn4l) dim=1280
  linear-component name=tdnn10l dim=320 $linear_opts input=Append(-3,0)
  relu-batchnorm-layer name=tdnn10 $opts input=Append(0,3) dim=1280
  linear-component name=lstm3l dim=320 $linear_opts input=Append(-3,0)
  fast-lstmp-layer name=lstm3 cell-dim=1536 recurrent-projection-dim=384 non-recurrent-projection-dim=384: delay=-3 dropout-proportion=0.0 $lstm_opts

  output-layer name=output input=lstm3  include-log-softmax=false $output_opts

  output-layer name=output-xent input=lstm3 learning-rate-factor=$learning_rate_factor $output_opts
EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
fi

if [ $stage -le 13 ]; then
  if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
    utils/create_split_dir.pl \
      /export/c0{1,2,5,7}/$USER/kaldi-data/egs/swbd-$(date +'%m_%d_%H_%M')/s5c/$dir/egs/storage $dir/egs/storage
  fi

  steps/nnet3/chain/train.py --stage $train_stage \
    --cmd "$decode_cmd" \
    --feat.online-ivector-dir $train_ivector_dir \
    --feat.cmvn-opts "--norm-means=false --norm-vars=false" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient 0.1 \
    --chain.l2-regularize 0.0 \
    --chain.apply-deriv-weights false \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --trainer.dropout-schedule $dropout_schedule \
    --trainer.num-chunk-per-minibatch 64,32 \
    --trainer.frames-per-iter 1500000 \
    --trainer.max-param-change 2.0 \
    --trainer.num-epochs 6 \
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
    --feat-dir $train_data_dir \
    --tree-dir $tree_dir \
    --lat-dir $lat_dir \
    --dir $dir  || exit 1;
fi


graph_dir=$dir/graph_tgsmall
if [ $stage -le 14 ]; then
  # Note: it might appear that this $lang directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.
  utils/mkgraph.sh --self-loop-scale 1.0 --remove-oov data/lang_test_tgsmall $dir $graph_dir
fi


iter_opts=
if [ ! -z $decode_iter ]; then
  iter_opts=" --iter $decode_iter "
fi
if [ $stage -le 15 ]; then
  rm $dir/.error 2>/dev/null || true
  for decode_set in test_clean test_other dev_clean dev_other; do
      (
      steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
          --nj $decode_nj --cmd "$decode_cmd" $iter_opts \
		  --extra-left-context $extra_left_context \
          --extra-right-context $extra_right_context \
          --extra-left-context-initial 0 \
          --extra-right-context-final 0 \
          --frames-per-chunk "$frames_per_chunk_primary" \
          --online-ivector-dir exp/nnet3${nnet3_affix}/ivectors_${decode_set}_hires \
          $graph_dir data/${decode_set}_hires $dir/decode_${decode_set}${decode_iter:+_$decode_iter}_tgsmall || exit 1
      steps/lmrescore.sh --cmd "$decode_cmd" --self-loop-scale 1.0 data/lang_test_{tgsmall,tgmed} \
          data/${decode_set}_hires $dir/decode_${decode_set}${decode_iter:+_$decode_iter}_{tgsmall,tgmed} || exit 1
      steps/lmrescore_const_arpa.sh \
          --cmd "$decode_cmd" data/lang_test_{tgsmall,tglarge} \
          data/${decode_set}_hires $dir/decode_${decode_set}${decode_iter:+_$decode_iter}_{tgsmall,tglarge} || exit 1
      steps/lmrescore_const_arpa.sh \
          --cmd "$decode_cmd" data/lang_test_{tgsmall,fglarge} \
          data/${decode_set}_hires $dir/decode_${decode_set}${decode_iter:+_$decode_iter}_{tgsmall,fglarge} || exit 1
      ) || touch $dir/.error &
  done
  wait
  if [ -f $dir/.error ]; then
    echo "$0: something went wrong in decoding"
    exit 1
  fi
fi
