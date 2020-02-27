#!/usr/bin/env bash

. ./cmd.sh

 # This step interpolates a small RNNLM (with weight 0.25) with the 4-gram LM.
steps/rnnlmrescore.sh \
  --N 100 --cmd "$decode_cmd" --inv-acwt 17 \
  0.25 data/lang_test_bd_fg data/local/rnnlm.h30.voc10k data/test_eval92 \
  exp/tri3b/decode_bd_tgpr_eval92_fg exp/tri3b/decode_bd_tgpr_eval92_fg_rnnlm30_0.25  \
  || exit 1;

steps/rnnlmrescore.sh \
  --N 100 --cmd "$decode_cmd" --inv-acwt 17 \
  0.5 data/lang_test_bd_fg data/local/rnnlm.h100.voc20k data/test_eval92 \
  exp/tri3b/decode_bd_tgpr_eval92_fg exp/tri3b/decode_bd_tgpr_eval92_fg_rnnlm100_0.5 \
  || exit 1;

steps/rnnlmrescore.sh \
  --N 100 --cmd "$decode_cmd" --inv-acwt 17 \
  0.5 data/lang_test_bd_fg data/local/rnnlm.h200.voc30k data/test_eval92 \
  exp/tri3b/decode_bd_tgpr_eval92_fg exp/tri3b/decode_bd_tgpr_eval92_fg_rnnlm200_0.5 \
  || exit 1;

steps/rnnlmrescore.sh \
  --N 100 --cmd "$decode_cmd" --inv-acwt 17 \
  0.5 data/lang_test_bd_fg data/local/rnnlm.h300.voc40k data/test_eval92 \
  exp/tri3b/decode_bd_tgpr_eval92_fg exp/tri3b/decode_bd_tgpr_eval92_fg_rnnlm300_0.5 \
  || exit 1;

steps/rnnlmrescore.sh \
  --N 1000 --cmd "$decode_cmd" --inv-acwt 17 \
  0.5 data/lang_test_bd_fg data/local/rnnlm.h300.voc40k data/test_eval92 \
  exp/tri3b/decode_bd_tgpr_eval92_fg exp/tri3b/decode_bd_tgpr_eval92_fg_rnnlm300_0.5_N1000 

dir=exp/tri3b/decode_bd_tgpr_eval92_fg_rnnlm300_0.75_N1000
rm -rf $dir
cp -r exp/tri3b/decode_bd_tgpr_eval92_fg_rnnlm300_0.5_N1000 $dir
steps/rnnlmrescore.sh \
  --stage 7 --N 1000 --cmd "$decode_cmd" --inv-acwt 17 \
  0.75 data/lang_test_bd_fg data/local/rnnlm.h300.voc40k data/test_eval92 \
  exp/tri3b/decode_bd_tgpr_eval92_fg $dir

dir=exp/tri3b/decode_bd_tgpr_eval92_fg_rnnlm300_0.75
rm -rf $dir
cp -r exp/tri3b/decode_bd_tgpr_eval92_fg_rnnlm300_0.5 $dir
steps/rnnlmrescore.sh \
  --stage 7 --N 100 --cmd "$decode_cmd" --inv-acwt 17 \
  0.75 data/lang_test_bd_fg data/local/rnnlm.h300.voc40k data/test_eval92 \
  exp/tri3b/decode_bd_tgpr_eval92_fg $dir

dir=exp/tri3b/decode_bd_tgpr_eval92_fg_rnnlm300_0.25
rm -rf $dir
cp -r exp/tri3b/decode_bd_tgpr_eval92_fg_rnnlm300_0.5 $dir
steps/rnnlmrescore.sh \
  --stage 7 --N 100 --cmd "$decode_cmd" --inv-acwt 17 \
  0.25 data/lang_test_bd_fg data/local/rnnlm.h300.voc40k data/test_eval92 \
  exp/tri3b/decode_bd_tgpr_eval92_fg $dir

steps/rnnlmrescore.sh \
  --N 10 --cmd "$decode_cmd" --inv-acwt 17 \
  0.5 data/lang_test_bd_fg data/local/rnnlm.h300.voc40k data/test_eval92 \
  exp/tri3b/decode_bd_tgpr_eval92_fg exp/tri3b/decode_bd_tgpr_eval92_fg_rnnlm300_0.5_N10 \
  || exit 1;

