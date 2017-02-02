#!/bin/bash

echo -n "# System               "
for x in $*; do   printf "% 10s" $x;   done
echo

echo -n "# WER on train_dev(tg) "
for x in $*; do
  wer=$(grep WER exp/nnet3/${x}_sp/decode_train_dev_hires_sw1_tg/wer_* | utils/best_wer.sh | awk '{print $2}')
  printf "% 10s" $wer
done
echo

echo -n "# WER on train_dev(fg) "
for x in $*; do
  wer=$(grep WER exp/nnet3/${x}_sp/decode_train_dev_hires_sw1_fsh_fg/wer_* | utils/best_wer.sh | awk '{print $2}')
  printf "% 10s" $wer
done
echo

echo -n "# WER on eval2000(tg)  "
for x in $*; do
  wer=$(grep Sum exp/nnet3/${x}_sp/decode_eval2000_hires_sw1_tg/score*/*ys | grep -v swbd | utils/best_wer.sh | awk '{print $2}')
  printf "% 10s" $wer
done
echo

echo -n "# WER on eval2000(fg)  "
for x in $*; do
  wer=$(grep Sum exp/nnet3/${x}_sp/decode_eval2000_hires_sw1_fsh_fg/score*/*ys | grep -v swbd | utils/best_wer.sh | awk '{print $2}')
  printf "% 10s" $wer
done
echo

echo -n "# Final train prob     "
for x in $*; do
  prob=$(grep log-likelihood exp/nnet3/${x}_sp/log/compute_prob_train.combined.log | awk '{print $8}')
  printf "% 10s" $prob
done
echo

echo -n "# Final valid prob     "
for x in $*; do
  prob=$(grep log-likelihood exp/nnet3/${x}_sp/log/compute_prob_valid.combined.log | awk '{print $8}')
  printf "% 10s" $prob
done
echo

