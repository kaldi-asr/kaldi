#!/bin/bash

# this script is used for comparing decoding results between systems.
# e.g. local/chain/compare_wer_general.sh exp/chain/tdnn_7b exp/chain/tdnn_lstm_1a

echo "# $0 $*";  # print command line.

echo -n "# System                     "
for x in $*; do   printf " % 20s" $x;   done
echo


echo -n "# WER on dev_aspire (fg)     "
for dirname in $*; do
  wer=$(grep -H Sum $dirname/decode*dev_aspire*fg/score*/*/ctm*.sys | utils/best_wer.sh | awk '{print $2}')
  printf "% 19s" $wer
done
echo

echo -n "# Final train prob           "
for dirname in $*; do
  prob=$(grep Overall $dirname/log/compute_prob_train.final.log | grep -v xent | awk '{print $8}')
  printf "% 19.3f" $prob
done
echo

echo -n "# Final valid prob           "
for dirname in $*; do
  prob=$(grep Overall $dirname/log/compute_prob_valid.final.log | grep -v xent | awk '{print $8}')
  printf "% 19.3f" $prob
done
echo

echo -n "# Final train prob (xent)    "
for dirname in $*; do
  prob=$(grep Overall $dirname/log/compute_prob_train.final.log | grep -w xent | awk '{print $8}')
  printf "% 19.3f" $prob
done
echo

echo -n "# Final valid prob (xent)    "
for dirname in $*; do
  prob=$(grep Overall $dirname/log/compute_prob_valid.final.log | grep -w xent | awk '{print $8}')
  printf "% 19.4f" $prob
done
echo

echo -n "# Num-parameters             "
for dirname in $*; do
  num_params=$(grep num-parameters $dirname/log/progress.1.log | awk '{print $2}')
  printf "% 19d" $num_params
done
echo
