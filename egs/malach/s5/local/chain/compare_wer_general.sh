#!/usr/bin/env bash

echo -n "System               "
for x in $*; do   printf " % 10s" $x;   done
echo

#for d in exp/chain_cleaned/tdnn*/decode_*; do grep Sum $d/*sc*/*ys | utils/best_wer.sh; done|grep eval_hires


echo -n "WER on dev  "
for x in $*; do
  wer=$(grep Sum exp/chain_cleaned/${x}/decode_dev/*sc*/*ys | utils/best_wer.sh | awk '{print $2}')
  printf "% 10s" $wer
done
echo

echo -n "Rescore with lstm 1a  "
for x in $*; do
  wer=$(grep Sum exp/chain_cleaned/${x}/decode_dev*tdnn_1a/*sc*/*ys | utils/best_wer.sh | awk '{print $2}')
  printf "% 10s" $wer
done
echo

echo -n "Rescore with lstm 1b  "
for x in $*; do
  wer=$(grep Sum exp/chain_cleaned/${x}/decode_dev*tdnn_1b/*sc*/*ys | utils/best_wer.sh | awk '{print $2}')
  printf "% 10s" $wer
done
echo

echo -n "Rescore with lstm bs_1a  "
for x in $*; do
  wer=$(grep Sum exp/chain_cleaned/${x}/decode_dev*tdnn_bs_1a/*sc*/*ys | utils/best_wer.sh | awk '{print $2}')
  printf "% 10s" $wer
done
echo

echo -n "Final train prob     "
for x in $*; do
  if [[ "${x}" != *online* ]]; then  
  prob=$(grep Overall exp/chain_cleaned/${x}/log/compute_prob_train.final.log | grep -v xent | awk '{print $8}')
  printf "% 10s" $prob
  fi
done
echo

echo -n "Final valid prob     "
for x in $*; do
  if [[ "${x}" != *online* ]]; then
  prob=$(grep Overall exp/chain_cleaned/${x}/log/compute_prob_valid.final.log | grep -v xent | awk '{print $8}')
  printf "% 10s" $prob
  fi
done
echo

echo -n "Final train prob (xent)    "
for x in $*; do
  if [[ "${x}" != *online* ]]; then
  prob=$(grep Overall exp/chain_cleaned/${x}/log/compute_prob_train.final.log | grep -w xent | awk '{print $8}')
  printf "% 10s" $prob
  fi
done
echo

echo -n "Final valid prob (xent)    "
for x in $*; do
  if [[ "${x}" != *online* ]]; then
  prob=$(grep Overall exp/chain_cleaned/${x}/log/compute_prob_valid.final.log | grep -w xent | awk '{print $8}')
  printf "% 10s" $prob
  fi
done
echo
