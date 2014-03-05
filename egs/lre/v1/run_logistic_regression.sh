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

awk '{print $2}' data/train/utt2lang  | sort -u | \
  awk '{print $1, NR-1}' >  exp/ivectors_train/languages.txt


log=exp/ivectors_train/log/logistic_regression.log

model=exp/ivectors_train/logistic_regression
train_ivectors=exp/ivectors_train/ivector.scp
classes="ark:utils/sym2int.pl -f 2 exp/ivectors_train/languages.txt data/train/utt2lang|"


utils/sym2int.pl -f 2 exp/ivectors_train/languages.txt data/train/utt2lang | \
  awk '{print $2}' | sort -n | uniq -c | \
  awk 'BEGIN{printf(" [ ");} {printf("%s ", 1.0/$1); } END{print(" ]"); }' \
   >exp/ivectors_train/inv_priors.vec



. path.sh

logistic-regression-train --config=$config scp:$train_ivectors \
                          "$classes" $model 2>$log 

#( logistic-regression-train --config=$config scp:$train_ivectors \
#                          "$classes" - | \
# logistic-regression-copy --scale-priors=exp/ivectors_train/inv_priors.vec - $model ) 2>$log

trials="utils/sym2int.pl -f 2 exp/ivectors_train/languages.txt data/train/utt2lang|"
scores="|utils/int2sym.pl -f 2 exp/ivectors_train/languages.txt  >exp/ivectors_train/train_scores"

logistic-regression-eval $model scp:$train_ivectors ark,t:exp/ivectors_train/posteriors
logistic-regression-eval $model "ark:$trials" scp:$train_ivectors "$scores"

logistic-regression-eval $model scp:$train_ivectors ark,t:- | \
  awk '{max=$3; argmax=3; for(f=3;f<NF;f++) { if ($f>max) { max=$f; argmax=f; }}  print $1, (argmax - 3); }' | \
  utils/int2sym.pl -f 2 exp/ivectors_train/languages.txt >exp/ivectors_train/output

# note: we treat the language as a sentence; it happens that the WER/SER
# corresponds to the recognition error rate.
compute-wer --text ark:data/train/utt2lang ark:exp/ivectors_train/output

# It perfectly classifies the training data:
#%WER 0.00 [ 0 / 10173, 0 ins, 0 del, 0 sub ]
#%SER 0.00 [ 0 / 10173 ]
#Scored 10173 sentences, 0 not present in hyp.


logistic-regression-eval $model scp:exp/ivectors_test/ivector.scp ark,t:- | \
  awk '{max=$3; argmax=3; for(f=3;f<NF;f++) { if ($f>max) { max=$f; argmax=f; }}  print $1, (argmax - 3); }' | \
  utils/int2sym.pl -f 2 exp/ivectors_train/languages.txt >exp/ivectors_test/output


# someone needs to extend this to run on the dev data.

compute-wer --text ark:data/test/utt2lang ark:exp/ivectors_test/output

#compute-wer --text ark:data/test/utt2lang ark:exp/ivectors_test/output
#compute-wer --text ark:data/test/utt2lang ark:exp/ivectors_test/output 
#%WER 2.97 [ 119 / 4000, 0 ins, 0 del, 119 sub ]
#%SER 2.97 [ 119 / 4000 ]
#Scored 4000 sentences, 0 not present in hyp.

