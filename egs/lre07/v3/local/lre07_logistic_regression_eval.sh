#!/bin/bash
# Copyright  2014   David Snyder
# Apache 2.0.
#
# Calculates the 3s, 10s, and 30s error rates and C_avgs
# on the LRE07 General Language Recognition closed-set
# using the logistic regression model passed in as an argument.
# Detailed results such as the probability of misses for individual
# languages are computed in local/lre07_results.

. cmd.sh
. path.sh
set -e

model=$1

mkdir -p local/lre07_results
lre07dir=local/lre07_results

# Compute the posterior probabilities for all durations (3s, 10s, and 30s),
# as well as the target and nontarget files.
test_ivectors="ark:ivector-normalize-length \
         scp:exp/ivectors_lre07/ivector.scp ark:- |";
logistic-regression-eval $model "$test_ivectors" \
  ark,t:exp/ivectors_lre07/posteriors

local/lre07_targets.pl exp/ivectors_lre07/posteriors data/lre07/utt2lang \
  exp/ivectors_train/languages.txt $lre07dir/targets \
  $lre07dir/nontargets>/dev/null

# Create the the score (eg, targets.scr) file.
local/score_lre07.v01d.pl -t $lre07dir/targets -n $lre07dir/nontargets

# Compute the posterior probabilities for each duration, as well as
# the target and nontarget files.
for dur in "3" "10" "30"; do
  utils/filter_scp.pl -f 0 data/lre07/"$dur"sec \
    exp/ivectors_lre07/ivector.scp > \
    exp/ivectors_lre07/ivector_"$dur"sec.scp
  test_ivectors="ark:ivector-normalize-length \
         scp:exp/ivectors_lre07/ivector_"$dur"sec.scp ark:- |";

  logistic-regression-eval $model "$test_ivectors" \
    ark,t:exp/ivectors_lre07/posteriors_"$dur"sec

  local/lre07_targets.pl exp/ivectors_lre07/posteriors_"$dur"sec \
    <(utils/filter_scp.pl -f 0 data/lre07/"$dur"sec data/lre07/utt2lang) \
    exp/ivectors_train/languages.txt \
    "$lre07dir"/targets_"$dur"sec "$lre07dir"/nontargets_"$dur"sec>/dev/null
  local/score_lre07.v01d.pl -t "$lre07dir"/targets_"$dur"sec -n \
    "$lre07dir"/nontargets_"$dur"sec>/dev/null
done

printf '% 15s' 'Duration (sec):'
for dur in "avg" "3" "10" "30"; do
  printf '% 7s' $dur;
done
echo

printf '% 15s' 'ER (%):'

# Get the overall classification and then individual duration error rates.
er=$(compute-wer --text ark:<(lid/remove_dialect.pl data/lre07/utt2lang) \
  ark:exp/ivectors_lre07/output 2>/dev/null | grep "WER" | awk '{print $2 }')
printf '% 7.2f' $er

for dur in "3" "10" "30"; do
  er=$(compute-wer --text ark:<(utils/filter_scp.pl -f 0 \
    data/lre07/"$dur"sec data/lre07/utt2lang | lid/remove_dialect.pl -) \
    ark:<(utils/filter_scp.pl -f 0 data/lre07/"$dur"sec \
      exp/ivectors_lre07/output) \
    2>/dev/null | grep "WER" | awk '{print $2 }')
    printf '% 7.2f' $er
done
echo

printf '% 15s' 'C_avg (%):'

# Get the overall C_avg and then C_avgs for the individual durations.
cavg=$(tail -n 1 $lre07dir/targets.scr \
     | awk '{print 100*$4 }')
printf '% 7.2f' $cavg

for dur in "3" "10" "30"; do
  cavg=$(tail -n 1 $lre07dir/targets_${dur}sec.scr \
       | awk '{print 100.0*$4 }')
  printf '% 7.2f' $cavg
done
echo
# Duration (sec):    avg      3     10     30
#         ER (%):  33.04  53.21  29.55  16.37
#      C_avg (%):  17.65  29.53  15.64   7.79
