#!/bin/bash
# Copyright 2018  Tsinghua University (Author: Zhiyuan Tang)
# Apache 2.0.


. ./cmd.sh
. ./path.sh

n=8 # parallel jobs

set -eu


###### Bookmark: basic preparation ######

# prepare training set and test set in data/{train,test},
# both contain at least wav.scp, utt2lang, spk2utt and utt2spk,
# spk2utt/utt2spk could be fake, e.g. the utt-id is just the spk-id

# prepare trials in data/test,
# by comparing each utt in test/ to all language labels in train/
local/prepare_trials.py data/train data/test
trials=data/test/trials


###### Bookmark: feature and alignment generation ######

# produce Fbank in data/fbank/{train,test}
rm -rf data/fbank && mkdir -p data/fbank && cp -r data/{train,test} data/fbank
for x in train test; do
  steps/make_fbank.sh --nj $n --cmd "$train_cmd" data/fbank/$x
done

## if vad needed, uncomment following lines to make MFCC with energy and vad
## and do alignment, dnn training and scoring with true vad option.
#rm -rf data/mfcc && mkdir -p data/mfcc && cp -r data/{train,test} data/mfcc
#for x in train test; do
#  steps/make_mfcc.sh --nj $n --cmd "$train_cmd" data/mfcc/$x
#  sid/compute_vad_decision.sh --nj $n --cmd "$train_cmd" data/mfcc/$x data/mfcc/$x/log data/mfcc/$x/data
#  cp data/mfcc/$x/vad.scp data/fbank/$x/vad.scp
#done

# prepare lang int id alignment per utt for making training egs
local/lang_ali.py -novad data/fbank/train exp/lang_ali


###### Bookmark: dnn training and scoring ######

# tdnn
local/nnet3/run_tdnn.sh data/fbank/train exp/lang_ali exp/tdnn
local/nnet3/run_score.sh --nj $n --cmd "$train_cmd" exp/tdnn data/fbank/test exp/lang_ali exp/tdnn_scores

# lstm
local/nnet3/run_lstm.sh data/fbank/train exp/lang_ali exp/lstm
local/nnet3/run_score.sh --nj $n --cmd "$train_cmd" \
  --frames-per-chunk 40 --extra-left-context 40 --extra-right-context 0 \
  --extra-left-context-initial 0 --extra-right-context-final 0 \
  exp/lstm data/fbank/test exp/lang_ali exp/lstm_scores

## TODO
## ptn
#local/nnet3/run_ptn.sh data/fbank/train exp/lang_ali exp/ptn
#local/nnet3/run_score.sh --nj $n --cmd "$train_cmd"  \
#  --frames-per-chunk 40 --extra-left-context 40 --extra-right-context 0 \
#  --extra-left-context-initial 0 --extra-right-context-final 0 \
#  exp/ptn data/fbank/test exp/lang_ali exp/ptn_scores

# print frame and utt level eer and cavg
for i in tdnn lstm; do
#  eer=`compute-eer <(python local/nnet3/prepare_for_eer.py $trials exp/${i}_scores/*frame) 2> /dev/null`
#  printf "%15s %5.2f \n" "$i frame level eer%:" $eer
#  cavg=`python local/compute_cavg.py -matrix $trials exp/${i}_scores/*frame`
#  printf "%15s %7.4f \n" "$i frame level cavg:" $cavg
  eer=`compute-eer <(python local/nnet3/prepare_for_eer.py $trials exp/${i}_scores/*utt) 2> /dev/null`
  printf "%15s %5.2f \n" "$i utt level eer%:" $eer
  cavg=`python local/compute_cavg.py -matrix $trials exp/${i}_scores/*utt`
  printf "%15s %7.4f \n" "$i utt level cavg:" $cavg
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
#        for i in tdnn lstm; do
#          eer=`compute-eer <(python local/nnet3/prepare_for_eer.py $trials exp/${i}_scores/*frame) 2> /dev/null`
#          printf "%15s %5.2f \n" "$i frame level eer%:" $eer
#          cavg=`python local/compute_cavg.py -matrix $trials exp/${i}_scores/*frame`
#          printf "%15s %7.4f \n" "$i frame level cavg:" $cavg
#          eer=`compute-eer <(python local/nnet3/prepare_for_eer.py $trials exp/${i}_scores/*utt) 2> /dev/null`
#          printf "%15s %5.2f \n" "$i utt level eer%:" $eer
#          cavg=`python local/compute_cavg.py -matrix $trials exp/${i}_scores/*utt`
#          printf "%15s %7.4f \n" "$i utt level cavg:" $cavg
#        done
#      scanned="${scanned}_$p$q"
#      fi
#    fi
#  done
#done


exit 0
