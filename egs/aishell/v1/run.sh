#!/bin/bash
# Copyright 2017 Beijing Shell Shell Tech. Co. Ltd. (Authors: Hui Bu)
#           2017 Jiayu Du
#           2017 Chao Li
#           2017 Xingyu Na
#           2017 Bengu Wu
#           2017 Hao Zheng
# Apache 2.0

# This is a shell script that we demonstrate speech recognition using AIShell-1 data.
# it's recommended that you run the commands one by one by copying and pasting into the shell.
# See README.txt for more info on data required.
# Results (EER) are inline in comments below

data=/export/a05/xna/data
data_url=www.openslr.org/resources/33

. ./cmd.sh
. ./path.sh

set -e # exit on error

local/download_and_untar.sh $data $data_url data_aishell
local/download_and_untar.sh $data $data_url resource_aishell

# Data Preparation
local/aishell_data_prep.sh $data/data_aishell/wav $data/data_aishell/transcript

# Now make MFCC  features.
# mfccdir should be some place with a largish disk where you
# want to store MFCC features.
mfccdir=mfcc
for x in train test; do
  steps/make_mfcc.sh --cmd "$train_cmd" --nj 10 data/$x exp/make_mfcc/$x $mfccdir
  sid/compute_vad_decision.sh --nj 10 --cmd "$train_cmd" data/$x exp/make_mfcc/$x $mfccdir
  utils/fix_data_dir.sh data/$x
done

# train diag ubm
sid/train_diag_ubm.sh --nj 10 --cmd "$train_cmd" --num-threads 16 \
  data/train 1024 exp/diag_ubm_1024

#train full ubm
sid/train_full_ubm.sh --nj 10 --cmd "$train_cmd" data/train \
  exp/diag_ubm_1024 exp/full_ubm_1024

#train ivector
sid/train_ivector_extractor.sh --cmd "$train_cmd --mem 10G" \
  --num-iters 5 exp/full_ubm_1024/final.ubm data/train \
  exp/extractor_1024

#extract ivector
sid/extract_ivectors.sh --cmd "$train_cmd" --nj 10 \
  exp/extractor_1024 data/train exp/ivector_train_1024

#train plda
$train_cmd exp/ivector_train_1024/log/plda.log \
  ivector-compute-plda ark:data/train/spk2utt \
  'ark:ivector-normalize-length scp:exp/ivector_train_1024/ivector.scp  ark:- |' \
  exp/ivector_train_1024/plda

#split the test to enroll and eval
mkdir -p data/test/enroll data/test/eval
cp data/test/{spk2utt,feats.scp,vad.scp} data/test/enroll
cp data/test/{spk2utt,feats.scp,vad.scp} data/test/eval
local/split_data_enroll_eval.py data/test/utt2spk  data/test/enroll/utt2spk  data/test/eval/utt2spk
trials=data/test/aishell_speaker_ver.lst
local/produce_trials.py data/test/eval/utt2spk $trials
utils/fix_data_dir.sh data/test/enroll
utils/fix_data_dir.sh data/test/eval

#extract enroll ivector
sid/extract_ivectors.sh --cmd "$train_cmd" --nj 10 \
  exp/extractor_1024 data/test/enroll  exp/ivector_enroll_1024
#extract eval ivector
sid/extract_ivectors.sh --cmd "$train_cmd" --nj 10 \
  exp/extractor_1024 data/test/eval  exp/ivector_eval_1024

#compute plda score
$train_cmd exp/ivector_eval_1024/log/plda_score.log \
  ivector-plda-scoring --num-utts=ark:exp/ivector_enroll_1024/num_utts.ark \
  exp/ivector_train_1024/plda \
  ark:exp/ivector_enroll_1024/spk_ivector.ark \
  "ark:ivector-normalize-length scp:exp/ivector_eval_1024/ivector.scp ark:- |" \
  "cat '$trials' | awk '{print \\\$2, \\\$1}' |" exp/trials_out

#compute eer
awk '{print $3}' exp/trials_out | paste - $trials | awk '{print $1, $4}' | compute-eer -

# Result
# Scoring against data/test/aishell_speaker_ver.lst
# Equal error rate is 0.140528%, at threshold -12.018

exit 0
