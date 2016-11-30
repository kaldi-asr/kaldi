#!/bin/bash

echo $0 $*

echo -n "System               "
for x in $*; do   printf "% 10s" " $(basename $x)";   done
echo

echo -n "WER on dev(orig)    "
for x in $*; do
  wer=$(grep Sum $x/decode_dev/score*/*ys | utils/best_wer.sh | awk '{print $2}')
  printf "% 10s" $wer
done
echo

echo -n "WER on dev(rescored)"
for x in $*; do
  wer=$(grep Sum $x/decode_dev_rescore/score*/*ys | utils/best_wer.sh | awk '{print $2}')
  printf "% 10s" $wer
done
echo

echo -n "WER on test(orig)    "
for x in $*; do
  wer=$(grep Sum $x/decode_test/score*/*ys | utils/best_wer.sh | awk '{print $2}')
  printf "% 10s" $wer
done
echo

echo -n "WER on test(rescored)"
for x in $*; do
  wer=$(grep Sum $x/decode_test_rescore/score*/*ys | utils/best_wer.sh | awk '{print $2}')
  printf "% 10s" $wer
done
echo


echo -n "Final train prob     "
for x in $*; do
  prob=$(grep Overall $x/log/compute_prob_train.final.log | grep -v xent | awk '{printf("%.4f", $8)}')
  printf "% 10s" $prob
done
echo

echo -n "Final valid prob     "
for x in $*; do
  prob=$(grep Overall $x/log/compute_prob_valid.final.log | grep -v xent | awk '{printf("%.4f", $8)}')
  printf "% 10s" $prob
done
echo

echo -n "Final train prob (xent)"
for x in $*; do
  prob=$(grep Overall $x/log/compute_prob_train.final.log | grep -w xent | awk '{printf("%.4f", $8)}')
  printf "% 10s" $prob
done
echo

echo -n "Final valid prob (xent)"
for x in $*; do
  prob=$(grep Overall $x/log/compute_prob_valid.final.log | grep -w xent | awk '{printf("%.4f", $8)}')
  printf "% 10s" $prob
done
echo
