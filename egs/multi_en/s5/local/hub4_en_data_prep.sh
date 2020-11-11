#!/usr/bin/env bash

# 1996/1997 English Broadcast News training data preparation (HUB4)

# Copyright  2017  Xiaohui Zhang
#            2017  Vimal Manohar
# Apache 2.0.
if [ $# != 4 ]; then
   echo "Arguments should be a list of HUB4 directories, see ../run.sh for example."
   exit 1;
fi

hub4_96_train_transcripts=$1
hub4_96_train_speech=$2
hub4_97_train_transcripts=$3
hub4_97_train_speech=$4

. ./path.sh # Needed for KALDI_ROOT
###############################################################################
# Prepare 1996 English Broadcast News Train (HUB4)
###############################################################################
local/hub4_96_data_prep.sh \
  $hub4_96_train_transcripts \
  $hub4_96_train_speech \
  data/local/train_bn96

###############################################################################
# Prepare 1996 English Broadcast News Train (HUB4)
###############################################################################
local/hub4_97_data_prep.sh \
  $hub4_97_train_transcripts \
  $hub4_97_train_speech \
  data/local/train_bn97

###############################################################################
# Format 1996 English Broadcast News Train (HUB4)
###############################################################################
mkdir -p data/hub4_en/train_bn96

local/hub4_format_data.pl \
  data/local/train_bn96/audio.list data/local/train_bn96/transcripts.txt \
  data/hub4_en/train_bn96 || exit 1

mv data/hub4_en/train_bn96/text data/hub4_en/train_bn96/text.unnorm
local/hub4_normalize_bn96_transcripts.pl "<noise>" "<spoken_noise>" \
  < data/hub4_en/train_bn96/text.unnorm > data/hub4_en/train_bn96/text

###############################################################################
# Format 1997 English Broadcast News Train (HUB4)
###############################################################################
mkdir -p data/hub4_en/train_bn97

local/hub4_format_data.pl \
  data/local/train_bn97/audio.list data/local/train_bn97/transcripts.txt \
  data/hub4_en/train_bn97 || exit 1

mv data/hub4_en/train_bn97/text data/hub4_en/train_bn97/text.unnorm
local/hub4_normalize_bn97_transcripts.pl "<noise>" "<spoken_noise>" \
  < data/hub4_en/train_bn97/text.unnorm > data/hub4_en/train_bn97/text

# Combine 1996/1997 BN data
utils/combine_data.sh data/hub4_en/train data/hub4_en/train_bn96 data/hub4_en/train_bn97
