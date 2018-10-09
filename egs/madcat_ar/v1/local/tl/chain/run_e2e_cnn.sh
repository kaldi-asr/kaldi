#!/bin/bash
# Copyright    2017  Hossein Hadian

# This script does end2end chain training (i.e. from scratch)

# ./local/chain/compare_wer.sh exp/chain/e2e_cnn_1a/
# System                      e2e_cnn_1a
# WER                             19.30
# CER                              5.72
# Final train prob              -0.0734
# Final valid prob              -0.0607
# Final train prob (xent)
# Final valid prob (xent)
# Parameters                      3.30M

# steps/info/chain_dir_info.pl exp/chain/e2e_cnn_1a/
# exp/chain/e2e_cnn_1a/: num-iters=24 nj=3..15 num-params=3.3M dim=56->292 combine=-0.060->-0.060 (over 1) logprob:train/valid[15,23,final]=(-0.122,-0.143,-0.073/-0.105,-0.132,-0.061)

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
common_egs_dir=
frames_per_iter=1000000
cmvn_opts="--norm-means=false --norm-vars=false"
train_set=train
lang_decode=data/lang

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
  # Use our special topology... note that later on may have to tune this
  # topology.
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
  common1="required-time-offsets= height-offsets=-2,-1,0,1,2 num-filters-out=36"
  common2="required-time-offsets= height-offsets=-2,-1,0,1,2 num-filters-out=70"
  common3="required-time-offsets= height-offsets=-1,0,1 num-filters-out=70"

  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig
  input dim=56 name=input
  conv-relu-batchnorm-layer name=cnn1 height-in=56 height-out=56 time-offsets=-3,-2,-1,0,1,2,3 $common1
  conv-relu-batchnorm-layer name=cnn2 height-in=56 height-out=28 time-offsets=-2,-1,0,1,2 $common1 height-subsample-out=2
  conv-relu-batchnorm-layer name=cnn3 height-in=28 height-out=28 time-offsets=-4,-2,0,2,4 $common2
  conv-relu-batchnorm-layer name=cnn4 height-in=28 height-out=28 time-offsets=-4,-2,0,2,4 $common2
  conv-relu-batchnorm-layer name=cnn5 height-in=28 height-out=14 time-offsets=-4,-2,0,2,4 $common2 height-subsample-out=2
  conv-relu-batchnorm-layer name=cnn6 height-in=14 height-out=14 time-offsets=-4,0,4 $common3
  conv-relu-batchnorm-layer name=cnn7 height-in=14 height-out=14 time-offsets=-4,0,4 $common3
  relu-batchnorm-layer name=tdnn1 input=Append(-4,0,4) dim=$tdnn_dim
  relu-batchnorm-layer name=tdnn2 input=Append(-4,0,4) dim=$tdnn_dim
  relu-batchnorm-layer name=tdnn3 input=Append(-4,0,4) dim=$tdnn_dim
  ## adding the layers for chain branch
  relu-batchnorm-layer name=prefinal-chain dim=$tdnn_dim target-rms=0.5
  output-layer name=output include-log-softmax=false dim=$num_targets max-change=1.5
EOF

  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs
fi

if [ $stage -le 3 ]; then
  # no need to store the egs in a shared storage because we always
  # remove them. Anyway, it takes only 5 minutes to generate them.

  steps/nnet3/chain/e2e/train_e2e.py --stage $train_stage \
    --cmd "$cmd" \
    --feat.cmvn-opts "$cmvn_opts" \
    --chain.leaky-hmm-coefficient 0.1 \
    --chain.l2-regularize 0.00005 \
    --chain.apply-deriv-weights false \
    --egs.dir "$common_egs_dir" \
    --egs.stage $get_egs_stage \
    --egs.opts "--num_egs_diagnostic 100 --num_utts_subset 400" \
    --chain.frame-subsampling-factor 4 \
    --chain.alignment-subsampling-factor 4 \
    --chain.lm-opts="--ngram-order=2 --no-prune-ngram-order=1 --num-extra-lm-states=1000" \
    --trainer.add-option="--optimization.memory-compression-level=2" \
    --trainer.num-chunk-per-minibatch $minibatch_size \
    --trainer.frames-per-iter 2000000 \
    --trainer.num-epochs 2 \
    --trainer.optimization.momentum 0 \
    --trainer.optimization.num-jobs-initial 3 \
    --trainer.optimization.num-jobs-final 16 \
    --trainer.optimization.initial-effective-lrate 0.001 \
    --trainer.optimization.final-effective-lrate 0.0001 \
    --trainer.optimization.shrink-value 1.0 \
    --trainer.max-param-change 2.0 \
    --cleanup.remove-egs true \
    --feat-dir data/${train_set} \
    --tree-dir $treedir \
    --dir $dir  || exit 1;
fi

if [ $stage -le 4 ]; then
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

if [ $stage -le 5 ]; then
  frames_per_chunk=$(echo $chunk_width | cut -d, -f1)
  steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
    --nj $nj --cmd "$cmd" \
    $dir/graph data/test $dir/decode_test || exit 1;
fi

echo "Done. Date: $(date). Results:"
local/chain/compare_wer.sh $dir
