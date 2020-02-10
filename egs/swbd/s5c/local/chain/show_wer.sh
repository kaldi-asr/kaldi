#!/usr/bin/env bash

for l in $*; do
  grep WER exp/chain/tdnn_${l}_sp/decode_train_dev_sw1_tg/wer_* | utils/best_wer.sh
done
for l in $*; do
  grep WER exp/chain/tdnn_${l}_sp/decode_train_dev_sw1_fsh_fg/wer_* | utils/best_wer.sh
done
for l in $*; do
  grep Sum exp/chain/tdnn_${l}_sp/decode_eval2000_sw1_tg/score*/*ys | grep -v swbd | utils/best_wer.sh
done
for l in $*; do
  grep Sum exp/chain/tdnn_${l}_sp/decode_eval2000_sw1_fsh_fg/score*/*ys | grep -v swbd | utils/best_wer.sh
done
