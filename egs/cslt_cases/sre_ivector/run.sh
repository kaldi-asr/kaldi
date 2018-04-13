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

# corpus and trans directory
thchs=/nfs/public/materials/data/thchs30-openslr

# you can obtain the database by uncommting the following lines
# [ -d $thchs ] || mkdir -p $thchs
# echo "downloading THCHS30 at $thchs ..."
# local/download_and_untar.sh $thchs  http://www.openslr.org/resources/18 data_thchs30
# local/download_and_untar.sh $thchs  http://www.openslr.org/resources/18 resource
# local/download_and_untar.sh $thchs  http://www.openslr.org/resources/18 test-noise

# generate text, wav.scp, utt2pk, spk2utt in data/{train,test}
local/thchs-30_data_prep.sh $thchs/data_thchs30
# randomly select 1000 utts from data/test as enrollment in data/enroll
# using rest utts in data/test for test
utils/subset_data_dir.sh data/test 1000 data/enroll
utils/filter_scp.pl --exclude data/enroll/wav.scp data/test/wav.scp > data/test/wav.scp.rest
mv data/test/wav.scp.rest data/test/wav.scp
utils/fix_data_dir.sh data/test

# prepare trials in data/test
local/prepare_trials.py data/enroll data/test
trials=data/test/trials


###### Bookmark: feature extraction ######

# produce MFCC feature with energy and its vad in data/mfcc/{train,enroll,test}
rm -rf data/mfcc && mkdir -p data/mfcc && cp -r data/{train,enroll,test} data/mfcc
for x in train enroll test; do
  steps/make_mfcc.sh --nj $n --cmd "$train_cmd" data/mfcc/$x
  sid/compute_vad_decision.sh --nj $n --cmd "$train_cmd" data/mfcc/$x data/mfcc/$x/log data/mfcc/$x/data
done


###### Bookmark: i-vector training ######

# reduce the amount of training data for UBM, num of utts depends on the total
utils/subset_data_dir.sh data/mfcc/train 3000 data/mfcc/train_3k
utils/subset_data_dir.sh data/mfcc/train 6000 data/mfcc/train_6k

# train UBM
sid/train_diag_ubm.sh --cmd "$train_cmd" --nj $n --num-threads 2 \
  data/mfcc/train_3k $num_gauss $exp/diag_ubm
sid/train_full_ubm.sh --cmd "$train_cmd" --nj $n \
  data/mfcc/train_6k $exp/diag_ubm $exp/full_ubm

# train i-vetor extractor
sid/train_ivector_extractor.sh --cmd "$train_cmd" --nj $n \
  --num-processes 1 --num-threads 1 \
  --ivector-dim $ivector_dim --num-iters 5 \
  $exp/full_ubm/final.ubm data/mfcc/train $exp/extractor


###### Bookmark: i-vector extraction ######

sid/extract_ivectors.sh --cmd "$train_cmd" --nj $n \
  $exp/extractor data/mfcc/train $exp/ivectors_train

sid/extract_ivectors.sh --cmd "$train_cmd" --nj $n \
  $exp/extractor data/mfcc/enroll $exp/ivectors_enroll

sid/extract_ivectors.sh --cmd "$train_cmd" --nj $n \
  $exp/extractor data/mfcc/test $exp/ivectors_test


###### Bookmark: cosine scoring ######

# basic cosine scoring on i-vectors
local/cosine_scoring.sh data/mfcc/enroll data/mfcc/test \
  $exp/ivectors_enroll $exp/ivectors_test $trials $exp/scores

# cosine scoring after reducing the i-vector dim with LDA
local/lda_scoring.sh data/mfcc/train data/mfcc/enroll data/mfcc/test \
  $exp/ivectors_train $exp/ivectors_enroll $exp/ivectors_test $trials $exp/scores

# cosine scoring after reducing the i-vector dim with PLDA
local/plda_scoring.sh data/mfcc/train data/mfcc/enroll data/mfcc/test \
  $exp/ivectors_train $exp/ivectors_enroll $exp/ivectors_test $trials $exp/scores

# print eer
for i in cosine lda plda; do
  eer=`compute-eer <(python local/prepare_for_eer.py $trials $exp/scores/${i}_scores) 2> /dev/null`
  printf "%15s %5.2f \n" "$i eer:" $eer
done


exit 0
