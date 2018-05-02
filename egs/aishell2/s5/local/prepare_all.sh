#!/bin/bash
# Copyright 2018 AIShell-Foundation(Authors:Jiayu DU, Xingyu NA, Bengu WU, Hao ZHENG)
#           2018 Beijing Shell Shell Tech. Co. Ltd. (Author: Hui BU)
# Apache 2.0

corpus=
stage=1

. ./cmd.sh
. ./path.sh

if [ $# -ne 1 ]; then
	echo "prepare.sh <corpus-data-dir>"
	echo " e.g prepare.sh /home/data/corpus/AISHELL-2/iOS/data"
	exit 1;
fi

corpus=$1

# lexicon and word segmentation tool
if [ $stage -le 1 ]; then
	local/prepare_dict.sh data/local/dict || exit 1;
fi

# wav.scp, text(word-segmented), utt2spk, spk2utt
if [ $stage -le 2 ]; then
	local/prepare_data.sh $corpus data/local/dict data/train || exit 1;
fi

# L
if [ $stage -le 4 ]; then
	utils/prepare_lang.sh --position-dependent-phones false \
		data/local/dict "<UNK>" data/local/lang data/lang || exit 1;
fi

# arpa LM
if [ $stage -le 5 ]; then
	local/train_lms.sh || exit 1;
fi

# G compilation, check LG composition
if [ $stage -le 6 ]; then
	utils/format_lm.sh data/lang data/local/lm/3gram-mincount/lm_unpruned.gz \
		data/local/dict/lexicon.txt data/lang_test || exit 1;
fi

echo "local/prepare.sh succeeded"
exit 0;
