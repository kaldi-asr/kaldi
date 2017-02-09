#!/bin/bash

# Prints a table makes it easy to compare WER and objective values across nnet3
# and chain training runs

echo -n "System               "
for x in "$@"; do   printf "% 10s" $x;   done
echo

echo -n "WER on dev(tg) "
for x in "$@"; do
  wer=$(grep WER ${x}/decode_dev/wer_* | utils/best_wer.sh | awk '{print $2}')
  printf "% 10s" $wer
done
echo

echo -n "WER on test(tg)  "
for x in "$@"; do
  wer=$(grep WER ${x}/decode_test/wer_* | utils/best_wer.sh | awk '{print $2}')
  printf "% 10s" $wer
done
echo

echo -n "Final train prob     "
for x in "$@"; do
  prob=$(grep Overall ${x}/log/compute_prob_train.final.log | grep -v xent | awk '{printf("%.4f", $8)}')
  printf "% 10s" $prob
done
echo

echo -n "Final valid prob     "
for x in "$@"; do
  prob=$(grep Overall ${x}/log/compute_prob_valid.final.log | grep -v xent | awk '{printf("%.4f", $8)}')
  printf "% 10s" $prob
done
echo

echo -n "Final train prob (xent)    "
for x in "$@"; do
  prob=$(grep Overall ${x}/log/compute_prob_train.final.log | grep -w xent | awk '{printf("%.4f", $8)}')
  printf "% 10s" $prob
done
echo

echo -n "Final valid prob (xent)    "
for x in "$@"; do
  prob=$(grep Overall ${x}/log/compute_prob_valid.final.log | grep -w xent | awk '{printf("%.4f", $8)}')
  printf "% 10s" $prob
done
echo
