#!/bin/bash
# Copyright  2014   David Snyder,  Daniel Povey
# Apache 2.0.
#
# An in progress example script for training and evaluating
# using logistic regression.

. cmd.sh
. path.sh
set -e

base=exp/phonotactics
config=conf/logistic-regression.conf

awk '{print $2}' <(lid/remove_dialect.pl $base/utt2lang_train) | sort -u | \
  awk '{print $1, NR-1}' >  $base/languages.txt

model=$base/logistic_regression
model_rebalanced=$base/logistic_regression_rebalanced
train_vectors="scp:$base/train_feats.scp";
test_vectors="scp:$base/test_feats.scp";
test30_vectors="scp:$base/test_feats30s.scp";
test3_vectors="$base/test_feats.scp";
test10_vectors="$base/test_feats.scp";
classes="ark:lid/remove_dialect.pl $base/utt2lang_train \
         | utils/sym2int.pl -f 2 $base/languages.txt - |"

lid/balance_priors_to_test.pl \
    <(lid/remove_dialect.pl <(utils/filter_scp.pl -f 1 \
        $base/train_feats.scp $base/utt2lang_train)) \
    <(lid/remove_dialect.pl $base/utt2lang_test) \
    $base/languages.txt \
    0.80 \
    $base/priors.vec

logistic-regression-train --config=$config "$train_vectors" \
                          "$classes" $model \
   2>$base/log/logistic_regression.log


 logistic-regression-copy --scale-priors=$base/priors.vec \
   $model $model_rebalanced

trials="lid/remove_dialect.pl $base/utt2lang_train \
        | utils/sym2int.pl -f 2 $base/languages.txt -|"
scores="|utils/int2sym.pl -f 2 $base/languages.txt  \
        >$base/train_scores"

logistic-regression-eval $model "$train_vectors" \
  ark,t:$base/posteriors

cat $base/posteriors | \
  awk '{max=$3; argmax=3; for(f=3;f<NF;f++) { if ($f>max) 
                          { max=$f; argmax=f; }}  
                          print $1, (argmax - 3); }' | \
  utils/int2sym.pl -f 2 $base/languages.txt \
    >$base/output_train

# note: we treat the language as a sentence; it happens that the WER/SER
# corresponds to the recognition error rate.
compute-wer --mode=present --text ark:<(lid/remove_dialect.pl $base/utt2lang_train) \
  ark:$base/output_train

logistic-regression-eval $model_rebalanced $test_vectors ark,t:$base/posteriors

logistic-regression-eval $model_rebalanced $test_vectors ark,t:- | \
  awk '{max=$3; argmax=3; for(f=3;f<NF;f++) { if ($f>max) 
                          { max=$f; argmax=f; }}  
                          print $1, (argmax - 3); }' | \
  utils/int2sym.pl -f 2 $base/languages.txt >$base/output_test

compute-wer --text ark:<(lid/remove_dialect.pl $base//utt2lang_test) \
  ark:$base/output_test

