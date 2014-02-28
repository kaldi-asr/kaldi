#!/bin/bash
# Copyright  2014   David Snyder
# Apache 2.0.
#
# An in progress example script for training and evaluating
# using logistic regression.

. cmd.sh
. path.sh
set -e
mfccdir=`pwd`/mfcc
vaddir=`pwd`/mfcc


awk '{print $2}' data/train/utt2lang  | sort -u | \
  awk '{print $1, NR-1}' >  exp/ivectors_train/languages.txt


log=exp/ivectors_train/log/logistic_regression.log

model=exp/ivectors_train/logistic_regression
train_ivectors=exp/ivectors_train/ivector.scp
classes="ark:utils/sym2int.pl -f 2 exp/ivectors_train/languages.txt data/train/utt2lang|"

. path.sh
logistic-regression-train scp:$train_ivectors "$classes" $model 2>$log

posterior_output=posteriors
scores=posteriors

classes=exp/ivectors_train/trials
utils/utt2lang_to_utt2langint.py exp/ivectors_train/languages.txt \
    data/train/utt2lang $trials

logistic-regression-eval $model scp:$train_ivectors $posterior_output 2>$log
logistic-regression-eval $model ark:$trials scp:$train_ivectors $scores 2>$log

