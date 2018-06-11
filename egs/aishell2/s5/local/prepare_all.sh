#!/bin/bash

# Copyright 2018 AIShell-Foundation(Authors:Jiayu DU, Xingyu NA, Bengu WU, Hao ZHENG)
#           2018 Beijing Shell Shell Tech. Co. Ltd. (Author: Hui BU)
#           2018 Emotech LTD (Author: Xuechen LIU)
# Apache 2.0

trn_set=
dev_set=
tst_set=

stage=1

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

if [ $# -ne 3 ]; then
  echo "prepare_all.sh <corpus-train-dir> <corpus-dev-dir> <corpus-test-dir>"
  echo " e.g prepare_all.sh /data/AISHELL-2/iOS/train /data/AISHELL-2/iOS/dev /data/AISHELL-2/iOS/test"
  exit 1;
fi

trn_set=$1
dev_set=$2
tst_set=$3

# download DaCiDian raw resources, convert to Kaldi lexicon format
if [ $stage -le 1 ]; then
  local/prepare_dict.sh data/local/dict || exit 1;
fi

# wav.scp, text(word-segmented), utt2spk, spk2utt
if [ $stage -le 2 ]; then
  local/prepare_data.sh ${trn_set} data/local/dict data/local/train data/train || exit 1;
  local/prepare_data.sh ${dev_set} data/local/dict data/local/dev   data/dev   || exit 1;
  local/prepare_data.sh ${tst_set} data/local/dict data/local/test  data/test  || exit 1;
fi

# L
if [ $stage -le 3 ]; then
  utils/prepare_lang.sh --position-dependent-phones false \
    data/local/dict "<UNK>" data/local/lang data/lang || exit 1;
fi

# arpa LM
if [ $stage -le 4 ]; then
  local/train_lms.sh \
      data/local/dict/lexicon.txt \
      data/local/train/text \
      data/local/lm || exit 1;
fi

# G compilation, check LG composition
if [ $stage -le 5 ]; then
  utils/format_lm.sh data/lang data/local/lm/3gram-mincount/lm_unpruned.gz \
    data/local/dict/lexicon.txt data/lang_test || exit 1;
fi

echo "local/prepare_all.sh succeeded"
exit 0;

