#!/usr/bin/env bash

# Copyright 2021  Behavox (author: Hossein Hadian)
# Apache 2.0

stage=0
skip_decoding=false
train_set=train_so_20spk
test_sets="dev test_cv_mini"
ali_dir=
lores_train_data_dir=
train_data_dir=
tree_dir=
dir=
gmm=tri3b
nnet3_affix=
exp=exp
test_lang=
epochs=2
dim=1408
bdim=160
softmax=false
tree_opts=
leaves=5000
fpi=2000000
ilr=0.00025
flr=0.000025
l2reg1=0.01
l2reg2=0.05
save_interval=30

online_cmvn=false

# The rest are configs specific to this script.  Most of the parameters
# are just hardcoded at this level, in the commands below.
affix=   # affix for the TDNN directory name
tree_affix=
train_stage=-10
get_egs_stage=-10
decode_iter=

# training options
# training chunk-options
chunk_width=140,100,160
common_egs_dir=
xent_regularize=0.1

# training options
srand=0
remove_egs=true
reporting_email=
nj=20
dropout_schedule='0,0@0.20,0.3@0.50,0'

#decode options

# End configuration section.
echo "$0 $@"  # Print the command line for logging
fullcmd="$0 $@"

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

set -euo pipefail

if ! cuda-compiled; then
  cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi

if [ -z $ali_dir ]; then
  ali_dir=$exp/${gmm}_ali_${train_set}_sp
fi
tree_dir=$exp/chain/tree_sp${tree_affix:+_$tree_affix}
lang=data/lang_chain
lat_dir=$ali_dir
if [ -z $dir ]; then
  dir=$exp/chain/tdnn1a_noiv${affix}_sp
fi
if [ -z $train_data_dir ]; then
  train_data_dir=data/${train_set}_sp_hires
fi
if [ -z $lores_train_data_dir ]; then
  lores_train_data_dir=data/${train_set}_sp
fi

if [ -z $test_lang ]; then
  test_lang=$exp/lang_tg
fi
[ -z $tree_opts ] && tree_opts="--context-width=2 --central-position=1"

mkdir -p $dir
echo  $fullcmd >> $dir/cmd
echo $tree_dir > $dir/tree_dir

local/chain/run_perturb_common.sh --stage $stage --nj $nj --train-set $train_set --test-sets "$test_sets"
local/chain/run_chain_common.sh --stage $stage \
    --lores_train_data_dir $lores_train_data_dir \
    --train_data_dir $train_data_dir \
    --exp $exp \
    --gmm $exp/$gmm \
    --ali_lats_dir $ali_dir \
    --lang $exp/lang \
    --lang_chain $exp/lang_chain \
    --tree_dir $tree_dir \
    --leaves $leaves --tree-opts "$tree_opts" \
    --test_sets "$test_sets" \
    --nj $nj \
    --use-ivector false \
    --nnet3-affix "$nnet3_affix" || exit 1;

if [ $stage -le 13 ]; then
  echo "$0: creating neural net configs using the xconfig parser";

  num_targets=$(tree-info $tree_dir/tree |grep num-pdfs|awk '{print $2}')
  learning_rate_factor=$(echo "print (0.5/$xent_regularize)" | python)
  affine_opts="l2-regularize=$l2reg1 dropout-proportion=0.0 dropout-per-dim=true dropout-per-dim-continuous=true"
  tdnnf_opts="l2-regularize=$l2reg1 dropout-proportion=0.0 bypass-scale=0.66"
  linear_opts="l2-regularize=$l2reg1 orthonormal-constraint=-1.0"
  prefinal_opts="l2-regularize=$l2reg1"
  output_opts="l2-regularize=$l2reg2"

  mkdir -p $dir/configs

  cat <<EOF > $dir/configs/network.xconfig
  input dim=40 name=input

  delta-layer name=delta
  no-op-component name=input2

  relu-batchnorm-dropout-layer name=tdnn1 $affine_opts dim=$dim input=input2
  tdnnf-layer name=tdnnf2 $tdnnf_opts dim=$dim bottleneck-dim=$bdim time-stride=1
  tdnnf-layer name=tdnnf3 $tdnnf_opts dim=$dim bottleneck-dim=$bdim time-stride=1
  tdnnf-layer name=tdnnf4 $tdnnf_opts dim=$dim bottleneck-dim=$bdim time-stride=1
  tdnnf-layer name=tdnnf5 $tdnnf_opts dim=$dim bottleneck-dim=$bdim time-stride=0
  tdnnf-layer name=tdnnf6 $tdnnf_opts dim=$dim bottleneck-dim=$bdim time-stride=3
  tdnnf-layer name=tdnnf7 $tdnnf_opts dim=$dim bottleneck-dim=$bdim time-stride=3
  tdnnf-layer name=tdnnf8 $tdnnf_opts dim=$dim bottleneck-dim=$bdim time-stride=3
  tdnnf-layer name=tdnnf9 $tdnnf_opts dim=$dim bottleneck-dim=$bdim time-stride=3
  tdnnf-layer name=tdnnf10 $tdnnf_opts dim=$dim bottleneck-dim=$bdim time-stride=3
  tdnnf-layer name=tdnnf11 $tdnnf_opts dim=$dim bottleneck-dim=$bdim time-stride=3
  tdnnf-layer name=tdnnf12 $tdnnf_opts dim=$dim bottleneck-dim=$bdim time-stride=3
  tdnnf-layer name=tdnnf13 $tdnnf_opts dim=$dim bottleneck-dim=$bdim time-stride=3
  tdnnf-layer name=tdnnf14 $tdnnf_opts dim=$dim bottleneck-dim=$bdim time-stride=3
  tdnnf-layer name=tdnnf15 $tdnnf_opts dim=$dim bottleneck-dim=$bdim time-stride=3
  linear-component name=prefinal-l dim=256 $linear_opts

  prefinal-layer name=prefinal-chain input=prefinal-l $prefinal_opts big-dim=$dim small-dim=256
  output-layer name=output include-log-softmax=$softmax dim=$num_targets $output_opts

  prefinal-layer name=prefinal-xent input=prefinal-l $prefinal_opts big-dim=$dim small-dim=256
  output-layer name=output-xent dim=$num_targets learning-rate-factor=$learning_rate_factor $output_opts
EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
fi

if [ $stage -le 14 ]; then

  python3 steps/nnet3/chain/train.py --stage=$train_stage \
    --cmd="$decode_cmd" \
    --feat.cmvn-opts="--norm-means=false --norm-vars=false" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient=0.1 \
    --chain.l2-regularize=0.0 \
    --chain.apply-deriv-weights=false \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --trainer.dropout-schedule $dropout_schedule \
    --trainer.add-option="--optimization.memory-compression-level=2" \
    --trainer.srand=$srand \
    --trainer.max-param-change=2.0 \
    --trainer.num-epochs=$epochs \
    --trainer.frames-per-iter=$fpi \
    --trainer.optimization.num-jobs-initial=4 \
    --trainer.optimization.num-jobs-final=4 \
    --trainer.optimization.initial-effective-lrate $ilr \
    --trainer.optimization.final-effective-lrate $flr \
    --trainer.num-chunk-per-minibatch=128,64 \
    --egs.chunk-width=$chunk_width \
    --egs.dir="$common_egs_dir" \
    --egs.opts="--frames-overlap-per-eg 0 --constrained false" \
    --cleanup.remove-egs=$remove_egs \
    --feat-dir=$train_data_dir \
    --tree-dir=$tree_dir \
    --lat-dir=$lat_dir \
    --use-gpu=wait \
    --cleanup.preserve-model-interval=$save_interval \
    --dir=$dir  || exit 1;
fi

if $skip_decoding; then
  exit 0;
fi

if [ $stage -le 15 ]; then
  # Note: it's not important to give mkgraph.sh the lang directory with the
  # matched topology (since it gets the topology file from the model).
  utils/mkgraph.sh \
    --self-loop-scale 1.0 $test_lang \
    $tree_dir $tree_dir/graph_tg || exit 1;
fi

if [ $stage -le 16 ]; then
  frames_per_chunk=$(echo $chunk_width | cut -d, -f1)
  rm $dir/.error 2>/dev/null || true

  for data in $test_sets; do
    (
      nspk=$(wc -l <data/${data}_hires/spk2utt)
      if [ $nspk -gt $nj ]; then
        nspk=$nj
      fi
      steps/nnet3/decode.sh \
          --acwt 1.0 --post-decode-acwt 10.0 \
          --frames-per-chunk $frames_per_chunk \
          --nj $nspk --cmd "$decode_cmd"  --num-threads 4 \
          $tree_dir/graph_tg data/${data}_hires ${dir}/decode_tg_${data} || exit 1
    ) || touch $dir/.error &
  done
  wait
  [ -f $dir/.error ] && echo "$0: there was a problem while decoding" && exit 1
fi

exit 0;
