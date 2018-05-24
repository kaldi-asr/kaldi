#!/bin/bash
# Copyright 2018  Tsinghua University (Author: Zhiyuan Tang)
# Apache 2.0.


. ./cmd.sh
. ./path.sh

n=8 # parallel jobs
dvector_dim=400
exp=exp/dvector_tdnn_dim${dvector_dim}

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


###### Bookmark: feature and alignment generation ######

# produce Fbank in data/fbank/{train,enroll,test}
# MFCC with energy is needed for vad
rm -rf data/fbank && mkdir -p data/fbank && cp -r data/{train,enroll,test} data/fbank
rm -rf data/mfcc && mkdir -p data/mfcc && cp -r data/{train,enroll,test} data/mfcc
for x in train enroll test; do
  steps/make_fbank.sh --nj $n --cmd "$train_cmd" data/fbank/$x
  steps/make_mfcc.sh --nj $n --cmd "$train_cmd" data/mfcc/$x
  sid/compute_vad_decision.sh --nj $n --cmd "$train_cmd" data/mfcc/$x data/mfcc/$x/log data/mfcc/$x/data
  cp data/mfcc/$x/vad.scp data/fbank/$x/vad.scp
done

# prepare spk int id alignment per utt for making training egs
local/spk_ali.py -vad data/fbank/train exp/spk_ali


###### Bookmark: dnn training ######
local/nnet3/run_tdnn_raw.sh --vad true --dvector-dim ${dvector_dim} \
  data/fbank/train exp/spk_ali $exp/tdnn


###### Bookmark: d-vector extraction ######

# first set the last hidden layer as the output-node of the nnet
sed 's/input=output.log-softmax/input=tdnn6.renorm/g' \
  $exp/tdnn/final.raw > $exp/tdnn/final.raw.last_hid_out

local/extract_dvectors.sh --cmd "$train_cmd" --nj $n \
  $exp/tdnn data/fbank/train $exp/dvectors_train

local/extract_dvectors.sh --cmd "$train_cmd" --nj $n \
  $exp/tdnn data/fbank/enroll $exp/dvectors_enroll

local/extract_dvectors.sh --cmd "$train_cmd" --nj $n \
  $exp/tdnn data/fbank/test $exp/dvectors_test


###### Bookmark: cosine scoring ######

# basic cosine scoring on d-vectors
local/cosine_scoring.sh data/fbank/enroll data/fbank/test \
  $exp/dvectors_enroll $exp/dvectors_test $trials $exp/scores

# cosine scoring after reducing the d-vector dim with LDA
local/lda_scoring.sh data/fbank/train data/fbank/enroll data/fbank/test \
  $exp/dvectors_train $exp/dvectors_enroll $exp/dvectors_test $trials $exp/scores

# cosine scoring after reducing the d-vector dim with PLDA
local/plda_scoring.sh data/fbank/train data/fbank/enroll data/fbank/test \
  $exp/dvectors_train $exp/dvectors_enroll $exp/dvectors_test $trials $exp/scores

# print eer
for i in cosine lda plda; do
  eer=`compute-eer <(python local/prepare_for_eer.py $trials $exp/scores/${i}_scores) 2> /dev/null`
  printf "%15s %5.2f \n" "$i eer:" $eer

done

exit 0
