#!/bin/bash

# Copyright 2018 AIShell-Foundation(Authors:Jiayu DU, Xingyu NA, Bengu WU, Hao ZHENG)
#           2018 Beijing Shell Shell Tech. Co. Ltd. (Author: Hui BU)
# Apache 2.0

# AISHELL-2 provides:
#  * a Mandarin speech corpus (~1000hrs), free for non-commercial research/education use
#  * a baseline recipe setup for large scale Mandarin ASR system
# For more details, read $KALDI_ROOT/egs/aishell2/README.txt

# modify this to your AISHELL-2 training data path
# e.g:
# trn_set=/disk10/data/AISHELL-2/iOS/data
# dev_set=/disk10/data/AISHELL-2/iOS/dev
# tst_set=/disk10/data/AISHELL-2/iOS/test
trn_set=
dev_set=
tst_set=

nj=20
stage=1
mode=normal        # whether we wanna train simple or normal models. The 'normal' chain
                   # model includes pitch feats, i-vector and dropout, having stronger
                   # baseline results in terms of CER. Should be either 'normal' or 'simple'
gmm_stage=1

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

# check mode option
[ "$mode" != "normal" ] && [ "$mode" != "simple" ] && \
  echo "mode should be either 'normal' or 'simple'" && exit 0;

# prepare trn/dev/tst data, lexicon, lang etc
if [ $stage -le 1 ]; then
  local/prepare_all.sh ${trn_set} ${dev_set} ${tst_set} || exit 1;
fi

# GMM
if [ $stage -le 2 ]; then
  local/run_gmm.sh --nj $nj --stage $gmm_stage --mode $mode
fi

# nnet3 & chain
# so far _1d is the SoTA chain recipe for normal use. If better model get
# trained & tested it will hopefully be manually updated in later PRs
if [ $stage -le 3 ]; then
  [ $mode == "simple" ] && local/chain/run_tdnn.sh --nj $nj --stage 5 || \
    local/chain/tuning/run_tdnn_1d.sh --nj $nj --stage 5
fi

local/show_results.sh

exit 0;
