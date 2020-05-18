#!/bin/bash

# Copyright 2020 Audio, Speech and Language Processing Group (ASLP@NPU), Northwestern Polytechnical University (Authors: Zhuoyuan Yao, Xiong Wang, Jingyong Hou, Lei Xie)
#           2020 AIShell-Foundation (Author: Bengu WU) 
#           2020 Beijing Shell Shell Tech. Co. Ltd. (Author: Hui BU) 
# Apache 2.0


set -e

# configs for 'chain'
affix=kws
stage=0
train_stage=-10
get_egs_stage=-10
dir=exp/chain/tdnn_1b  
decode_iter=

# training options
num_epochs=4
initial_effective_lrate=0.001
final_effective_lrate=0.0001
max_param_change=2.0
final_layer_normalize_target=0.5
num_jobs_initial=2
num_jobs_final=2
nj=15
minibatch_size=128
dropout_schedule='0,0@0.20,0.3@0.50,0'
frames_per_eg=150,110,90
remove_egs=true
common_egs_dir=exp/chain/tdnn_1b_kws/egs
xent_regularize=0.1

# End configuration section.
echo "$0 $*"  # Print the command line for logging

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

dir=${dir}${affix:+_$affix}
train_set=fbank/train
test_sets="fbank/test"
ali_dir=exp/mono_ali
treedir=exp/chain/mono_cd_tree_sp
lang=data/lang_chain


if [ $stage -le 8 ]; then
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

if [ $stage -le 9 ]; then
  # Build a tree using our new topology. This is the critically different
  # step compared with other recipes.
  steps/nnet3/chain/build_tree.sh --frame-subsampling-factor 3 \
    --context-opts "--context-width=2 --central-position=1" \
    --cmd "$decode_cmd" 5000 data/$train_set $lang $ali_dir $treedir
fi

if [ $stage -le 10 ]; then
  echo "$0: creating neural net configs using the xconfig parser";
  feat_dim=$(feat-to-dim scp:data/${train_set}/feats.scp -)
  num_targets=$(tree-info $treedir/tree | grep num-pdfs | awk '{print $2}')
  learning_rate_factor=$(echo "print (0.5/$xent_regularize)" | python)
  opts="l2-regularize=0.002"
  linear_opts="orthonormal-constraint=1.0"
  output_opts="l2-regularize=0.0005 bottleneck-dim=8"

  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig
  input dim=$feat_dim name=input

  # please note that it is important to have input layer with the name=input
  # as the layer immediately preceding the fixed-affine-layer to enable
  # the use of short notation for the descriptor
  conv-relu-batchnorm-layer name=cnn1 height-in=71 height-out=71 time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=32
  linear-component name=cnnl1 dim=284 $linear_opts
  
  # the first splicing is moved before the lda layer, so no splicing here
  relu-batchnorm-layer name=tdnn1 dim=850
  relu-batchnorm-layer name=tdnn2 dim=850 input=Append(-1,0,2)
  relu-batchnorm-layer name=tdnn3 dim=850 input=Append(-3,0,3)
  relu-batchnorm-layer name=tdnn4 dim=850 input=Append(-7,0,2)
  relu-batchnorm-layer name=tdnn5 dim=850 input=Append(-3,0,3)
  relu-batchnorm-layer name=tdnn6 dim=850
  linear-component name=prefinal-l dim=256 $linear_opts 
  relu-batchnorm-layer name=prefinal-chain input=prefinal-l $opts dim=850 target-rms=0.5
  output-layer name=output include-log-softmax=false dim=$num_targets $output_opts max-change=1.5

  relu-batchnorm-layer name=prefinal-xent input=prefinal-l $opts dim=850 target-rms=0.5
  output-layer name=output-xent dim=$num_targets learning-rate-factor=$learning_rate_factor $output_opts max-change=1.5
EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
fi

if [ $stage -le 11 ]; then
  #if [[ $(hostname -f) == *.clsp.jhu.edu ]] && [ ! -d $dir/egs/storage ]; then
  #  utils/create_split_dir.pl \
  #   /export/b0{5,6,7,8}/$USER/kaldi-data/egs/aishell-$(date +'%m_%d_%H_%M')/s5c/$dir/egs/storage $dir/egs/storage
  #fi

  steps/nnet3/chain/train.py --stage $train_stage \
    --cmd "$cuda_cmd" \
    --feat.cmvn-opts "--norm-means=false --norm-vars=false" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient 0.1 \
    --chain.l2-regularize 0.00005 \
    --chain.apply-deriv-weights false \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --egs.cmd "$egs_cmd" \
	--egs.dir "$common_egs_dir" \
    --egs.stage $get_egs_stage \
    --egs.opts "--frames-overlap-per-eg 0" \
    --egs.chunk-width $frames_per_eg \
    --trainer.dropout-schedule $dropout_schedule \
    --trainer.num-chunk-per-minibatch $minibatch_size \
    --trainer.frames-per-iter 1500000 \
    --trainer.num-epochs $num_epochs \
    --trainer.optimization.num-jobs-initial $num_jobs_initial \
    --trainer.optimization.num-jobs-final $num_jobs_final \
    --trainer.optimization.initial-effective-lrate $initial_effective_lrate \
    --trainer.optimization.final-effective-lrate $final_effective_lrate \
    --trainer.max-param-change $max_param_change \
	--cleanup.remove-egs $remove_egs \
    --feat-dir data/${train_set} \
    --tree-dir $treedir \
    --lat-dir exp/datakws_mia_lats \
    --dir $dir  || exit 1;
fi

if [ $stage -le 12 ]; then
  # Note: it might appear that this $lang directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.
  utils/mkgraph.sh --self-loop-scale 1.0 data/lang_test $dir $dir/graph
fi

graph_dir=$dir/graph 
if [ $stage -le 13 ]; then
  for test_set in $test_sets; do
    nj=40
    steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
      --nj $nj --cmd "$decode_cmd" --stage 0 \
      $graph_dir data/${test_set} $dir/decode_test_new || exit 1;
  done
fi

echo "local/chain/run_tdnn.sh succeeded"
exit 0;
