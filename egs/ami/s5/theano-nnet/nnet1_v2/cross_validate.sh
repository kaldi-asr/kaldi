#!/bin/bash

. ./path.sh
. ./cmd.sh

cv_tool=theano-nnet/nnet1_v2/nnet_cross_validate.py
tool_opts=

feat_preprocess=

. utils/parse_options.sh || exit 1;

if [ $# != 3 ]; then
   echo "Usage: $0 <data-cv> <labels-cv> <nnet>"
   echo " e.g.: $0 data/cv labels_cv nnet"
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

data_cv_scp=$1
labels_cv=$2
nnet=$3

python $cv_tool \
  ${feat_preprocess:+ --feat-preprocess=$feat_preprocess} \
  ${tool_opts:+ $tool_opts} \
  $data_cv_scp $labels_cv $nnet || exit 1


