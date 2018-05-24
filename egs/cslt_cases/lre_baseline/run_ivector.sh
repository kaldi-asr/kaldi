#!/bin/bash
# Copyright 2018  Tsinghua University (Author: Zhiyuan Tang)
# Apache 2.0.


. ./cmd.sh
. ./path.sh

n=8 # parallel jobs

num_gauss=2048
ivector_dim=400
exp=exp/ivector_gauss${num_gauss}_dim${ivector_dim}

set -eu


###### Bookmark: basic preparation ######

# prepare training set and test set in data/{train,test},
# both contain at least wav.scp, utt2lang, spk2utt and utt2spk,
# spk2utt/utt2spk could be fake, e.g. the utt-id is just the spk-id

# prepare trials in data/test,
# by comparing each utt in test/ to all language labels in train/
local/prepare_trials.py data/train data/test
trials=data/test/trials


###### Bookmark: feature extraction ######

# produce MFCC feature with energy and its vad in data/mfcc/{train,test}
rm -rf data/mfcc && mkdir -p data/mfcc && cp -r data/{train,test} data/mfcc
for x in train test; do
  steps/make_mfcc.sh --nj $n --cmd "$train_cmd" data/mfcc/$x
  lid/compute_vad_decision.sh --nj $n --cmd "$train_cmd" data/mfcc/$x data/mfcc/$x/log data/mfcc/$x/data
done


###### Bookmark: i-vector training ######

# reduce the amount of training data for UBM, num of utts depends on the total
utils/subset_data_dir.sh data/mfcc/train 18000 data/mfcc/train_18k
utils/subset_data_dir.sh data/mfcc/train 36000 data/mfcc/train_36k

# train UBM
lid/train_diag_ubm.sh --cmd "$train_cmd" --nj $n --num-threads 2 \
  data/mfcc/train_18k $num_gauss $exp/diag_ubm
lid/train_full_ubm.sh --cmd "$train_cmd" --nj $n \
  data/mfcc/train_36k $exp/diag_ubm $exp/full_ubm

# train i-vetor extractor
lid/train_ivector_extractor.sh --cmd "$train_cmd" --nj $n \
  --num-processes 1 --num-threads 1 \
  --ivector-dim $ivector_dim --num-iters 5 \
  $exp/full_ubm/final.ubm data/mfcc/train $exp/extractor


###### Bookmark: i-vector extraction ######

lid/extract_ivectors.sh --cmd "$train_cmd" --nj $n \
  $exp/extractor data/mfcc/train $exp/ivectors_train

lid/extract_ivectors.sh --cmd "$train_cmd" --nj $n \
  $exp/extractor data/mfcc/test $exp/ivectors_test


###### Bookmark: cosine scoring ######

# basic cosine scoring on i-vectors
local/cosine_scoring.sh data/mfcc/train data/mfcc/test \
  $exp/ivectors_train $exp/ivectors_test $trials $exp/scores

# cosine scoring after reducing the i-vector dim with LDA
local/lda_scoring.sh data/mfcc/train data/mfcc/train data/mfcc/test \
  $exp/ivectors_train $exp/ivectors_train $exp/ivectors_test $trials $exp/scores

# cosine scoring after reducing the i-vector dim with PLDA
local/plda_scoring.sh data/mfcc/train data/mfcc/train data/mfcc/test \
  $exp/ivectors_train $exp/ivectors_train $exp/ivectors_test $trials $exp/scores

# print eer and cavg
for i in cosine lda plda; do
  eer=`compute-eer <(python local/prepare_for_eer.py $trials $exp/scores/${i}_scores) 2> /dev/null`
  printf "%15s %5.2f \n" "$i eer%:" $eer
  cavg=`python local/compute_cavg.py -pairs $trials $exp/scores/${i}_scores`
  printf "%15s %7.4f \n" "$i cavg:" $cavg
done

## print eer and cavg of different couples of languages
#lang_names="Kazak Tibet Uyghu ct-cn id-id ja-jp ko-kr ru-ru vi-vn zh-cn"
#scanned=""
#for p in $lang_names; do
#  for q in $lang_names; do
#    if [ $p != $q ]; then
#      if [[ ! $scanned =~ $q$p ]]; then
#        local/prepare_special_trials.py data/train "$p $q" data/test
#        trials=data/test/trials.${p}_${q}
#        echo "Results for language couple $p and $q:"
#        for i in cosine lda plda; do
#          eer=`compute-eer <(python local/prepare_for_eer.py $trials $exp/scores/${i}_scores) 2> /dev/null`
#          printf "%15s %5.2f \n" "$i eer%:" $eer
#          cavg=`python local/compute_cavg.py -pairs $trials $exp/scores/${i}_scores`
#          printf "%15s %7.4f \n" "$i cavg:" $cavg
#        done
#        scanned=${scanned}_$p$q
#      fi
#    fi
#  done
#done


###### Bookmark: SVM scoring ######
# TODO


exit 0
