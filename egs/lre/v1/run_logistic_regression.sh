#!/bin/bash
# Copyright  2014   David Snyder,  Daniel Povey
# Apache 2.0.
#
# An in progress example script for training and evaluating
# using logistic regression.

. cmd.sh
. path.sh
set -e
mfccdir=`pwd`/mfcc
vaddir=`pwd`/mfcc
config=conf/logistic-regression.conf

awk '{print $2}' <(utils/remove_dialect.pl data/train/utt2lang) | sort -u | \
  awk '{print $1, NR-1}' >  exp/ivectors_train/languages.txt


log=exp/ivectors_train/log/logistic_regression.log

model=exp/ivectors_train/logistic_regression
model_rebalanced=exp/ivectors_train/logistic_regression_rebalanced
train_ivectors=exp/ivectors_train/ivector.scp
classes="ark:utils/remove_dialect.pl data/train/utt2lang \
         | utils/sym2int.pl -f 2 exp/ivectors_train/languages.txt - |"

# An alternative prior.
#utils/sym2int.pl -f 2 exp/ivectors_train/languages.txt \
#  <(utils/remove_dialect.pl data/train/utt2lang) | \
#  awk '{print $2}' | sort -n | uniq -c | \
#  awk 'BEGIN{printf(" [ ");} {printf("%s ", 1.0/$1); } END{print(" ]"); }' \
#   >exp/ivectors_train/inv_priors.vec

# Create priors to rebalance the model. The following script rebalances
# the languages as count(lang_test) / (count(lang_test) + count(lang_train)).
utils/balance_priors_to_test.pl \
    <(utils/remove_dialect.pl data/train/utt2lang) \
    <(utils/remove_dialect.pl data/test/utt2lang) \
    exp/ivectors_train/languages.txt \
    exp/ivectors_train/priors.vec

logistic-regression-train --config=$config scp:$train_ivectors \
                          "$classes" $model 2>$log 

( logistic-regression-train --config=$config scp:$train_ivectors \
                          "$classes" - | \
 logistic-regression-copy --scale-priors=exp/ivectors_train/priors.vec - \
 $model_rebalanced ) 2>$log

trials="utils/remove_dialect.pl data/train/utt2lang \
        | utils/sym2int.pl -f 2 exp/ivectors_train/languages.txt -|"
scores="|utils/int2sym.pl -f 2 exp/ivectors_train/languages.txt  \
        >exp/ivectors_train/train_scores"

logistic-regression-eval $model scp:$train_ivectors \
  ark,t:exp/ivectors_train/posteriors

logistic-regression-eval $model "ark:$trials" scp:$train_ivectors "$scores"

logistic-regression-eval $model scp:$train_ivectors ark,t:- | \
  awk '{max=$3; argmax=3; for(f=3;f<NF;f++) { if ($f>max) 
                          { max=$f; argmax=f; }}  
                          print $1, (argmax - 3); }' | \
  utils/int2sym.pl -f 2 exp/ivectors_train/languages.txt \
    >exp/ivectors_train/output

# note: we treat the language as a sentence; it happens that the WER/SER
# corresponds to the recognition error rate.
compute-wer --text ark:<(utils/remove_dialect.pl data/train/utt2lang) \
  ark:exp/ivectors_train/output

# It perfectly classifies the training data:
#%WER 0.00 [ 0 / 10173, 0 ins, 0 del, 0 sub ]
#%SER 0.00 [ 0 / 10173 ]
#Scored 10173 sentences, 0 not present in hyp.


logistic-regression-eval $model_rebalanced \
  scp:exp/test/ivector.scp ark,t:- | \
  awk '{max=$3; argmax=3; for(f=3;f<NF;f++) { if ($f>max) 
                          { max=$f; argmax=f; }}  
                          print $1, (argmax - 3); }' | \
  utils/int2sym.pl -f 2 exp/ivectors_train/languages.txt >exp/test/output


# someone needs to extend this to run on the dev data.

compute-wer --text ark:<(utils/remove_dialect.pl data/test/utt2lang)\
  ark:exp/test/output
# compute-wer --text ark:/dev/fd/63 ark:exp/test/output 
# %WER 58.83 [ 3958 / 7527, 0 ins, 0 del, 3958 sub ]
# %SER 58.83 [ 3958 / 7527 ]
# Scored 7527 sentences, 0 not present in hyp.

