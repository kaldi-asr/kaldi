#!/bin/bash

. ./path.sh
. ./cmd.sh

train_tool=theano-nnet/nnet1_v2/train_1iter.py
tool_opts=

feat_preprocess=

. utils/parse_options.sh || exit 1;

if [ $# != 4 ]; then
   echo "Usage: $0 <data-train> <labels-train> <nnet-inp> <nnet-out>"
   echo " e.g.: $0 data/train labels_train nnet1 nnet2"
   echo ""
   echo " Training data : <data-train>,<ali-train> (for optimizing cross-entropy)"
   echo " Held-out data : <data-dev>,<ali-dev> (for learn-rate/model selection based on cross-entopy)"
   echo " note.: <ali-train>,<ali-dev> can point to same directory, or 2 separate directories."
   echo ""
   echo "main options (for others, see top of script file)"
   echo "  --config <config-file>   # config containing options"
   echo ""
   echo "  --copy-feats <bool>      # copy input features to /tmp (it's faster)"
   echo ""
   exit 1;

fi

feats_scp=$1
labels_tr=$2
mlp_best=$3
mlp_next=$4

python $train_tool \
  ${feat_preprocess:+ --feat-preprocess=$feat_preprocess} \
  ${tool_opts:+ $tool_opts} \
  $feats_scp $labels_tr $mlp_best $mlp_next || exit 1


