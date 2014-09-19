#!/bin/bash
# Copyright  2014   David Snyder,  Daniel Povey
# Apache 2.0.
#
# An in progress example script for training and evaluating
# using logistic regression.

. cmd.sh
. path.sh
set -e

train_dir=exp/ivectors_train
test_dir=exp/ivectors_lre07
model_dir=exp/ivectors_train
train_utt2lang=data/train/utt2lang
test_utt2lang=data/lre07/utt2lang
prior_scale=1.0
use_log_posteriors=true
conf=conf/logistic-regression.conf

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

mkdir -p $model_dir/log

awk '{print $2}' <(lid/remove_dialect.pl $train_utt2lang) | sort -u | \
  awk '{print $1, NR-1}' > $model_dir/languages.txt

model=$model_dir/logistic_regression
model_rebalanced=$model_dir/logistic_regression_rebalanced
train_ivectors="ark:ivector-normalize-length \
         scp:$train_dir/ivector.scp ark:- |";
test_ivectors="ark:ivector-normalize-length \
         scp:$test_dir/ivector.scp ark:- |";
classes="ark:lid/remove_dialect.pl $train_utt2lang \
         | utils/sym2int.pl -f 2 $model_dir/languages.txt - |"
# A uniform prior.
#utils/sym2int.pl -f 2 $model_dir/languages.txt \
#  <(lid/remove_dialect.pl $train_utt2lang) | \
#  awk '{print $2}' | sort -n | uniq -c | \
#  awk 'BEGIN{printf(" [ ");} {printf("%s ", 1.0/$1); } END{print(" ]"); }' \
#   >$model_dir/inv_priors.vec

# Create priors to rebalance the model. The following script rebalances
# the languages as ( count(lang_test) / count(lang_train) )^(prior_scale).
lid/balance_priors_to_test.pl \
    <(lid/remove_dialect.pl <(utils/filter_scp.pl -f 0 \
        $train_dir/ivector.scp $train_utt2lang)) \
    <(lid/remove_dialect.pl $test_utt2lang) \
    $model_dir/languages.txt \
    $prior_scale \
    $model_dir/priors.vec

logistic-regression-train --config=$conf "$train_ivectors" \
                          "$classes" $model \
   2>$model_dir/log/logistic_regression.log

logistic-regression-copy --scale-priors=$model_dir/priors.vec \
   $model $model_rebalanced

logistic-regression-eval --use-log-posteriors=$use_log_posteriors $model \
  "$train_ivectors" ark,t:$train_dir/posteriors

cat $train_dir/posteriors | \
  awk '{max=$3; argmax=3; for(f=3;f<NF;f++) { if ($f>max) 
                          { max=$f; argmax=f; }}  
                          print $1, (argmax - 3); }' | \
  utils/int2sym.pl -f 2 $model_dir/languages.txt \
    >$train_dir/output

# note: we treat the language as a sentence; it happens that the WER/SER
# corresponds to the recognition error rate.
compute-wer --mode=present --text ark:<(lid/remove_dialect.pl $train_utt2lang) \
  ark:$train_dir/output

# Evaluate on test data. Most likely a NIST LRE.
logistic-regression-eval --use-log-posteriors=$use_log_posteriors $model_rebalanced \
  "$test_ivectors" ark,t:$test_dir/posteriors

cat $test_dir/posteriors | \
  awk '{max=$3; argmax=3; for(f=3;f<NF;f++) { if ($f>max) 
                          { max=$f; argmax=f; }}  
                          print $1, (argmax - 3); }' | \
  utils/int2sym.pl -f 2 $model_dir/languages.txt \
    >$test_dir/output

compute-wer --text ark:<(lid/remove_dialect.pl $test_utt2lang) \
  ark:$test_dir/output
