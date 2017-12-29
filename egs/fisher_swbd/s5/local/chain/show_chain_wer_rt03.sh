#!/bin/bash

for l in $*; do
  grep Sum exp/chain/${1}/decode_rt03_fsh_sw1_tg/score*/rt03_hires.ctm.fsh.filt.sys | grep -v swbd | utils/best_wer.sh
done
for l in $*; do
  grep Sum exp/chain/${1}/decode_rt03_fsh_sw1_tg/score*/rt03_hires.ctm.filt.sys | grep -v swbd | utils/best_wer.sh
done
for l in $*; do
  grep Sum exp/chain/${1}/decode_rt03_fsh_sw1_tg/score*/rt03_hires.ctm.swbd.filt.sys  | utils/best_wer.sh
done
for l in $*; do
  grep Sum exp/chain/${1}/decode_rt03_fsh_sw1_fg/score*/rt03_hires.ctm.fsh.filt.sys | grep -v swbd | utils/best_wer.sh
done
for l in $*; do
  grep Sum exp/chain/${1}/decode_rt03_fsh_sw1_fg/score*/rt03_hires.ctm.filt.sys | grep -v swbd | utils/best_wer.sh
done
for l in $*; do
  grep Sum exp/chain/${1}/decode_rt03_fsh_sw1_fg/score*/rt03_hires.ctm.swbd.filt.sys | utils/best_wer.sh
done
