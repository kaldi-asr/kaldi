#!/bin/bash

# Copyright    2017  Hossein Hadian

# This script does end2end chain training (i.e. from scratch)
# local/chain/compare_wer.sh exp/chain/e2e_cnn_1a/
# System                          e2e_cnn_1a
#                                 score_basic  score_nomalized
# WER                             17.65        14.1
# WER (rescored)                  15.35        12.1
# CER                              3.97         4.0
# CER (rescored)                   3.41         3.5
# Final train prob                0.1402
# Final valid prob                0.1045

# steps/info/chain_dir_info.pl exp/chain/e2e_cnn_1a
#exp/chain/e2e_cnn_1a: num-iters=17 nj=6..12 num-params=3.0M dim=40->352 combine=0.110->0.110 (over 1) logprob:train/valid[10,16,final]=(0.135,0.134,0.140/0.098,0.099,0.105)

set -e
# configs for 'chain'
stage=0
nj=30
train_stage=-10
get_egs_stage=-10
affix=1a

# training options
tdnn_dim=450
minibatch_size=150=64,32/300=32,16/600=16,8/1200=8,4
cmvn_opts="--norm-means=false --norm-vars=false"
train_set=train
lang_decode=data/lang
lang_rescore=data/lang_rescore_6g
decode_e2e=true
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

lang=data/lang_e2e
treedir=exp/chain/e2e_monotree  # it's actually just a trivial tree (no tree building)
dir=exp/chain/e2e_cnn_${affix}

if [ $stage -le 0 ]; then
  # Create a version of the lang/ directory that has one state per phone in the
  # topo file. [note, it really has two states.. the first one is only repeated
  # once, the second one has zero or more repeats.]
  rm -rf $lang
  cp -r data/lang $lang
  silphonelist=$(cat $lang/phones/silence.csl) || exit 1;
  nonsilphonelist=$(cat $lang/phones/nonsilence.csl) || exit 1;
  steps/nnet3/chain/gen_topo.py $nonsilphonelist $silphonelist >$lang/topo
fi

if [ $stage -le 1 ]; then
  steps/nnet3/chain/e2e/prepare_e2e.sh --nj $nj --cmd "$cmd" \
                                       --shared-phones true \
                                       --type mono \
                                       data/$train_set $lang $treedir
  $cmd $treedir/log/make_phone_lm.log \
  cat data/$train_set/text \| \
    steps/nnet3/chain/e2e/text_to_phones.py data/lang \| \
    utils/sym2int.pl -f 2- data/lang/phones.txt \| \
    chain-est-phone-lm --num-extra-lm-states=500 \
                       ark:- $treedir/phone_lm.fst
fi

if [ $stage -le 2 ]; then
  echo "$0: creating neural net configs using the xconfig parser";
  num_targets=$(tree-info $treedir/tree | grep num-pdfs | awk '{print $2}')
  cnn_opts="l2-regularize=0.075"
  tdnn_opts="l2-regularize=0.075"
  output_opts="l2-regularize=0.1"
  common1="$cnn_opts required-time-offsets= height-offsets=-2,-1,0,1,2 num-filters-out=36"
  common2="$cnn_opts required-time-offsets= height-offsets=-2,-1,0,1,2 num-filters-out=70"
  common3="$cnn_opts required-time-offsets= height-offsets=-1,0,1 num-filters-out=70"

  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig
  input dim=40 name=input
  conv-relu-batchnorm-layer name=cnn1 height-in=40 height-out=40 time-offsets=-3,-2,-1,0,1,2,3 $common1
  conv-relu-batchnorm-layer name=cnn2 height-in=40 height-out=20 time-offsets=-2,-1,0,1,2 $common1 height-subsample-out=2
  conv-relu-batchnorm-layer name=cnn3 height-in=20 height-out=20 time-offsets=-4,-2,0,2,4 $common2
  conv-relu-batchnorm-layer name=cnn4 height-in=20 height-out=20 time-offsets=-4,-2,0,2,4 $common2
  conv-relu-batchnorm-layer name=cnn5 height-in=20 height-out=10 time-offsets=-4,-2,0,2,4 $common2 height-subsample-out=2
  conv-relu-batchnorm-layer name=cnn6 height-in=10 height-out=10 time-offsets=-4,0,4 $common3
  conv-relu-batchnorm-layer name=cnn7 height-in=10 height-out=10 time-offsets=-4,0,4 $common3
  relu-batchnorm-layer name=tdnn1 input=Append(-4,0,4) dim=$tdnn_dim $tdnn_opts
  relu-batchnorm-layer name=tdnn2 input=Append(-4,0,4) dim=$tdnn_dim $tdnn_opts
  relu-batchnorm-layer name=tdnn3 input=Append(-4,0,4) dim=$tdnn_dim $tdnn_opts
  ## adding the layers for chain branch
  relu-batchnorm-layer name=prefinal-chain dim=$tdnn_dim target-rms=0.5 $output_opts
  output-layer name=output include-log-softmax=false dim=$num_targets max-change=1.5 $output_opts
EOF

  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs
fi

if [ $stage -le 3 ]; then
  steps/nnet3/chain/e2e/train_e2e.py --stage $train_stage \
    --cmd "$cmd" \
    --feat.cmvn-opts "$cmvn_opts" \
    --chain.leaky-hmm-coefficient 0.1 \
    --chain.apply-deriv-weights true \
    --egs.stage $get_egs_stage \
    --egs.opts "--num_egs_diagnostic 100 --num_utts_subset 400" \
    --chain.frame-subsampling-factor 4 \
    --chain.alignment-subsampling-factor 4 \
    --trainer.add-option="--optimization.memory-compression-level=2" \
    --trainer.num-chunk-per-minibatch $minibatch_size \
    --trainer.frames-per-iter 1500000 \
    --trainer.num-epochs 3 \
    --trainer.optimization.momentum 0 \
    --trainer.optimization.num-jobs-initial 5 \
    --trainer.optimization.num-jobs-final 8 \
    --trainer.optimization.initial-effective-lrate 0.001 \
    --trainer.optimization.final-effective-lrate 0.0001 \
    --trainer.optimization.shrink-value 1.0 \
    --trainer.max-param-change 2.0 \
    --cleanup.remove-egs true \
    --feat-dir data/${train_set} \
    --tree-dir $treedir \
    --dir $dir  || exit 1;
fi

if [ $stage -le 4 ] && $decode_e2e; then
  # The reason we are using data/lang here, instead of $lang, is just to
  # emphasize that it's not actually important to give mkgraph.sh the
  # lang directory with the matched topology (since it gets the
  # topology file from the model).  So you could give it a different
  # lang directory, one that contained a wordlist and LM of your choice,
  # as long as phones.txt was compatible.

  utils/mkgraph.sh \
    --self-loop-scale 1.0 $lang_decode \
    $dir $dir/graph || exit 1;
fi

if [ $stage -le 5 ] && $decode_e2e; then
  frames_per_chunk=$(echo $chunk_width | cut -d, -f1)
  steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
    --nj 30 --cmd "$cmd" --beam 12 \
    $dir/graph data/test $dir/decode_test || exit 1;

  steps/lmrescore_const_arpa.sh --cmd "$cmd" $lang_decode $lang_rescore \
                                data/test $dir/decode_test{,_rescored} || exit 1

  echo "Done. Date: $(date). Results:"
  local/chain/compare_wer.sh $dir
fi
