#!/bin/bash
# Copyright  2014   David Snyder,  Daniel Povey
# Apache 2.0.
#
# An in progress example script for training and evaluating
# using logistic regression.

. cmd.sh
. path.sh
set -e

config=conf/logistic-regression.conf

awk '{print $2}' <(lid/remove_dialect.pl data/train/utt2lang) | sort -u | \
  awk '{print $1, NR-1}' >  exp/ivectors_train/languages.txt

model=exp/ivectors_train/logistic_regression
model_rebalanced=exp/ivectors_train/logistic_regression_rebalanced
train_ivectors="ark:ivector-normalize-length \
         scp:exp/ivectors_train/ivector.scp ark:- |";
classes="ark:lid/remove_dialect.pl data/train/utt2lang \
         | utils/sym2int.pl -f 2 exp/ivectors_train/languages.txt - |"

# An alternative prior.
#utils/sym2int.pl -f 2 exp/ivectors_train/languages.txt \
#  <(lid/remove_dialect.pl data/train/utt2lang) | \
#  awk '{print $2}' | sort -n | uniq -c | \
#  awk 'BEGIN{printf(" [ ");} {printf("%s ", 1.0/$1); } END{print(" ]"); }' \
#   >exp/ivectors_train/inv_priors.vec

# Create priors to rebalance the model. The following script rebalances
# the languages as count(lang_test) / (count(lang_test) + count(lang_train)).
lid/balance_priors_to_test.pl \
    <(lid/remove_dialect.pl <(utils/filter_scp.pl -f 0 \
        exp/ivectors_train/ivector.scp data/train/utt2lang)) \
    <(lid/remove_dialect.pl data/lre07/utt2lang) \
    exp/ivectors_train/languages.txt \
    exp/ivectors_train/priors.vec

logistic-regression-train --config=$config "$train_ivectors" \
                          "$classes" $model \
   2>exp/ivectors_train/log/logistic_regression.log


 logistic-regression-copy --scale-priors=exp/ivectors_train/priors.vec \
   $model $model_rebalanced

trials="lid/remove_dialect.pl data/train/utt2lang \
        | utils/sym2int.pl -f 2 exp/ivectors_train/languages.txt -|"
scores="|utils/int2sym.pl -f 2 exp/ivectors_train/languages.txt  \
        >exp/ivectors_train/train_scores"

logistic-regression-eval $model "$train_ivectors" \
  ark,t:exp/ivectors_train/posteriors

logistic-regression-eval $model "ark:$trials" "$train_ivectors" "$scores"

cat exp/ivectors_train/posteriors | \
  awk '{max=$3; argmax=3; for(f=3;f<NF;f++) { if ($f>max) 
                          { max=$f; argmax=f; }}  
                          print $1, (argmax - 3); }' | \
  utils/int2sym.pl -f 2 exp/ivectors_train/languages.txt \
    >exp/ivectors_train/output

# note: we treat the language as a sentence; it happens that the WER/SER
# corresponds to the recognition error rate.
compute-wer --mode=present --text ark:<(lid/remove_dialect.pl data/train/utt2lang) \
  ark:exp/ivectors_train/output

# %WER 4.73 [ 3389 / 71668, 0 ins, 0 del, 3389 sub ] [PARTIAL]
# %SER 4.73 [ 3389 / 71668 ]
# Scored 71668 sentences, 16 not present in hyp.
logistic-regression-eval $model_rebalanced \
  'ark:ivector-normalize-length scp:exp/ivectors_lre07/ivector.scp ark:- |' ark,t:- | \
  awk '{max=$3; argmax=3; for(f=3;f<NF;f++) { if ($f>max) 
                          { max=$f; argmax=f; }}  
                          print $1, (argmax - 3); }' | \
  utils/int2sym.pl -f 2 exp/ivectors_train/languages.txt >exp/ivectors_lre07/output

compute-wer --text ark:<(lid/remove_dialect.pl data/lre07/utt2lang) \
  ark:exp/ivectors_lre07/output

# %WER 33.04 [ 2487 / 7527, 0 ins, 0 del, 2487 sub ]
# %SER 33.04 [ 2487 / 7527 ]
# Scored 7527 sentences, 0 not present in hyp.

# General LR closed-set eval.
local/lre07_logistic_regression_eval.sh $model_rebalanced
#Duration (sec):    avg      3     10     30
#        ER (%):  33.04  53.21  29.55  16.37
#     C_avg (%):  17.65  29.53  15.64   7.79
