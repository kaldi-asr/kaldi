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

classes=exp/ivectors_sre08_train_short2/classes
utils/make_language_table.py data/train/utt2lang \
                             exp/ivectors_sre08_train_short2/languages.txt
utils/utt2lang_to_utt2langint.py exp/ivectors_sre08_train_short2/languages.txt \
    data/sre08_train_short2/utt2lang $classes

log=exp/ivectors_sre08_train_short2/log/logistic_regression.log

model=exp/ivectors_sre08_train_short2/logistic_regression
train_ivectors=exp/ivectors_sre08_train_short2/ivector.scp
                             
logistic-regression-train scp:$train_ivectors ark:$classes $model 2>$log

posterior_output=posteriors
scores=posteriors

classes=exp/ivectors_sre08_train_short2/trials
utils/utt2lang_to_utt2langint.py exp/ivectors_sre08_train_short2/languages.txt \
    data/sre08_train_short2/utt2lang $trials

logistic-regression-eval $model scp:$train_ivectors $posterior_output 2>$log
logistic-regression-eval $model ark:$trials scp:$train_ivectors $scores 2>$log

