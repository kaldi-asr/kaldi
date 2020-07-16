#!/usr/bin/env bash

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
gmm_stage=1

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

# prepare trn/dev/tst data, lexicon, lang etc
if [ $stage -le 1 ]; then
  local/prepare_all.sh ${trn_set} ${dev_set} ${tst_set} || exit 1;
fi

# GMM
if [ $stage -le 2 ]; then
  local/run_gmm.sh --nj $nj --stage $gmm_stage
fi

# chain
if [ $stage -le 3 ]; then
  local/chain/run_tdnn.sh --nj $nj
fi

local/show_results.sh

exit 0;
