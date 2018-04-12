#!/bin/bash

# Copyright 2016  Tsinghua University (Author: Dong Wang, Xuewei Zhang)
#           2018  Tsinghua University (Author: Zhiyuan Tang)
# Apache 2.0.

. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.
. ./path.sh


n=8 # parallel jobs

set -euo pipefail


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


###### Bookmark: language preparation ######

# prepare lexicon.txt, extra_questions.txt, nonsilence_phones.txt, optional_silence.txt, silence_phones.txt
# build a large lexicon that invovles words in both the training and decoding, all in data/dict
mkdir -p data/dict;
cp $thchs/resource/dict/{extra_questions.txt,nonsilence_phones.txt,optional_silence.txt,silence_phones.txt} data/dict && \
cat $thchs/resource/dict/lexicon.txt $thchs/data_thchs30/lm_word/lexicon.txt | \
grep -v '<s>' | grep -v '</s>' | sort -u > data/dict/lexicon.txt


###### Bookmark: language processing ######

# generate language stuff used for training
# also lexicon to L_disambig.fst for graph making in local/thchs-30_decode.sh
mkdir -p data/lang;
utils/prepare_lang.sh --position_dependent_phones false data/dict "<SPOKEN_NOISE>" data/local/lang data/lang

# format trained or provided language model to G.fst
# prepare things for graph making in local/thchs-30_decode.sh, not necessary for training
(
  mkdir -p data/graph;
  gzip -c $thchs/data_thchs30/lm_word/word.3gram.lm > data/graph/word.3gram.lm.gz
  utils/format_lm.sh data/lang data/graph/word.3gram.lm.gz $thchs/data_thchs30/lm_word/lexicon.txt data/graph/lang
)


###### Bookmark: feature extraction ######

# produce MFCC and Fbank features in data/{mfcc,fbank}/{train,test}
rm -rf data/mfcc && mkdir -p data/mfcc && cp -r data/{train,test} data/mfcc
rm -rf data/fbank && mkdir -p data/fbank && cp -r data/{train,test} data/fbank
for x in train test; do
  # make mfcc and fbank
  steps/make_mfcc.sh --nj $n --cmd "$train_cmd" data/mfcc/$x
  steps/make_fbank.sh --nj $n --cmd "$train_cmd" data/fbank/$x
  # compute cmvn
  steps/compute_cmvn_stats.sh data/mfcc/$x
  steps/compute_cmvn_stats.sh data/fbank/$x
done


###### Bookmark: GMM-HMM training & decoding ######

# monophone
steps/train_mono.sh --boost-silence 1.25 --nj $n --cmd "$train_cmd" data/mfcc/train data/lang exp/mono
# test monophone model
local/thchs-30_decode.sh --nj $n "steps/decode.sh" exp/mono data/mfcc &
# monophone ali
steps/align_si.sh --boost-silence 1.25 --nj $n --cmd "$train_cmd" data/mfcc/train data/lang exp/mono exp/mono_ali

# triphone
steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" 2000 10000 data/mfcc/train data/lang exp/mono_ali exp/tri1
# test tri1 model
local/thchs-30_decode.sh --nj $n "steps/decode.sh" exp/tri1 data/mfcc &
# triphone_ali
steps/align_si.sh --nj $n --cmd "$train_cmd" data/mfcc/train data/lang exp/tri1 exp/tri1_ali

# lda_mllt
steps/train_lda_mllt.sh --cmd "$train_cmd" --splice-opts "--left-context=3 --right-context=3" 2500 15000 data/mfcc/train data/lang exp/tri1_ali exp/tri2b
# test tri2b model
local/thchs-30_decode.sh --nj $n "steps/decode.sh" exp/tri2b data/mfcc &
# lda_mllt_ali
steps/align_si.sh  --nj $n --cmd "$train_cmd" --use-graphs true data/mfcc/train data/lang exp/tri2b exp/tri2b_ali

# sat
steps/train_sat.sh --cmd "$train_cmd" 2500 15000 data/mfcc/train data/lang exp/tri2b_ali exp/tri3b
# test tri3b model
local/thchs-30_decode.sh --nj $n "steps/decode_fmllr.sh" exp/tri3b data/mfcc &
# sat_ali
steps/align_fmllr.sh --nj $n --cmd "$train_cmd" data/mfcc/train data/lang exp/tri3b exp/tri3b_ali

# quick
steps/train_quick.sh --cmd "$train_cmd" 4200 40000 data/mfcc/train data/lang exp/tri3b_ali exp/tri4b
# test tri4b model
local/thchs-30_decode.sh --nj $n "steps/decode_fmllr.sh" exp/tri4b data/mfcc &
# quick_ali
steps/align_fmllr.sh --nj $n --cmd "$train_cmd" data/mfcc/train data/lang exp/tri4b exp/tri4b_ali


###### Bookmark: DNN training & decoding ######

# train tdnn model
tdnn_dir=exp/nnet3/tdnn
local/nnet3/run_tdnn.sh data/fbank/train exp/tri4b_ali $tdnn_dir

# decoding
graph_dir=exp/tri4b/graph_word # the same as gmm
steps/nnet3/decode.sh --nj 8 --cmd "$decode_cmd" $graph_dir data/fbank/test $tdnn_dir/decode_test_word


###### Bookmark: discriminative training & decoding ######

# mmi training
criterion=mmi # mmi, mpfe or smbr
local/nnet3/run_tdnn_discriminative.sh --criterion $criterion $tdnn_dir data/fbank/train

# decoding
steps/nnet3/decode.sh --nj 8 --cmd "$decode_cmd" $graph_dir data/fbank/test ${tdnn_dir}_$criterion/decode_test_word


exit 0
