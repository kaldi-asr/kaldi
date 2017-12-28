#!/bin/bash
for l in $*; do
  grep Sum exp/chain/${1}/decode_eval2000_fsh_sw1_tg/score*/eval2000_hires.ctm.callhm.filt.sys | grep -v swbd | utils/best_wer.sh
done
for l in $*; do
  grep Sum exp/chain/${1}/decode_eval2000_fsh_sw1_tg/score*/eval2000_hires.ctm.filt.sys | grep -v swbd | utils/best_wer.sh
done
for l in $*; do
  grep Sum exp/chain/${1}/decode_eval2000_fsh_sw1_tg/score*/eval2000_hires.ctm.swbd.filt.sys  | utils/best_wer.sh
done
for l in $*; do
  grep Sum exp/chain/${1}/decode_eval2000_fsh_sw1_fg/score*/eval2000_hires.ctm.callhm.filt.sys | grep -v swbd | utils/best_wer.sh
done
for l in $*; do
  grep Sum exp/chain/${1}/decode_eval2000_fsh_sw1_fg/score*/eval2000_hires.ctm.filt.sys | grep -v swbd | utils/best_wer.sh
done
for l in $*; do
  grep Sum exp/chain/${1}/decode_eval2000_fsh_sw1_fg/score*/eval2000_hires.ctm.swbd.filt.sys | utils/best_wer.sh
done
