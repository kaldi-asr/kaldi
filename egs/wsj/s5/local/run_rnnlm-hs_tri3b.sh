#!/bin/bash

. cmd.sh
 # This step interpolates a small RNNLM (with weight 0.15) with the 4-gram LM.
steps/rnnlmrescore.sh --rnnlm_ver rnnlm-hs-0.1b \
  --N 100 --cmd "$decode_cmd" --inv-acwt 17 \
  0.15 data/lang_test_bd_fg data/local/rnnlm-hs.h30.voc10k data/test_eval92 \
  exp/tri3b/decode_bd_tgpr_eval92_fg exp/tri3b/decode_bd_tgpr_eval92_fg_rnnlm-hs30_0.15  \
  || exit 1;

steps/rnnlmrescore.sh --rnnlm_ver rnnlm-hs-0.1b \
  --N 100 --cmd "$decode_cmd" --inv-acwt 17 \
  0.3 data/lang_test_bd_fg data/local/rnnlm-hs.h100.voc20k data/test_eval92 \
  exp/tri3b/decode_bd_tgpr_eval92_fg exp/tri3b/decode_bd_tgpr_eval92_fg_rnnlm-hs100_0.3 \
  || exit 1;

steps/rnnlmrescore.sh --rnnlm_ver rnnlm-hs-0.1b \
  --N 100 --cmd "$decode_cmd" --inv-acwt 17 \
  0.3 data/lang_test_bd_fg data/local/rnnlm-hs.h300.voc30k data/test_eval92 \
  exp/tri3b/decode_bd_tgpr_eval92_fg exp/tri3b/decode_bd_tgpr_eval92_fg_rnnlm-hs300_0.3 \
  || exit 1;

steps/rnnlmrescore.sh --rnnlm_ver rnnlm-hs-0.1b \
  --N 100 --cmd "$decode_cmd" --inv-acwt 17 \
  0.3 data/lang_test_bd_fg data/local/rnnlm-hs.h400.voc40k data/test_eval92 \
  exp/tri3b/decode_bd_tgpr_eval92_fg exp/tri3b/decode_bd_tgpr_eval92_fg_rnnlm-hs400_0.3 \
  || exit 1;

steps/rnnlmrescore.sh --rnnlm_ver rnnlm-hs-0.1b \
  --N 1000 --cmd "$decode_cmd" --inv-acwt 17 \
  0.3 data/lang_test_bd_fg data/local/rnnlm-hs.h400.voc40k data/test_eval92 \
  exp/tri3b/decode_bd_tgpr_eval92_fg exp/tri3b/decode_bd_tgpr_eval92_fg_rnnlm-hs400_0.3_N1000 

steps/rnnlmrescore.sh --rnnlm_ver rnnlm-hs-0.1b \
  --N 1000 --cmd "$decode_cmd" --inv-acwt 17 \
  0.3 data/lang_test_bd_fg data/local/rnnlm-hs.h400.voc40k data/test_eval92 \
  exp/tri3b/decode_bd_tgpr_eval92_fg exp/tri3b/decode_bd_tgpr_eval92_fg_rnnlm-hs400_0.3_N1000 \
  || exit 1;

dir=exp/tri3b/decode_bd_tgpr_eval92_fg_rnnlm-hs400_0.4_N1000
rm -rf $dir
cp -r exp/tri3b/decode_bd_tgpr_eval92_fg_rnnlm-hs400_0.3_N1000 $dir
steps/rnnlmrescore.sh --rnnlm_ver rnnlm-hs-0.1b \
  --stage 7 --N 1000 --cmd "$decode_cmd" --inv-acwt 17 \
  0.4 data/lang_test_bd_fg data/local/rnnlm-hs.h400.voc40k data/test_eval92 \
  exp/tri3b/decode_bd_tgpr_eval92_fg $dir

dir=exp/tri3b/decode_bd_tgpr_eval92_fg_rnnlm-hs400_0.4
rm -rf $dir
cp -r exp/tri3b/decode_bd_tgpr_eval92_fg_rnnlm-hs400_0.3 $dir
steps/rnnlmrescore.sh --rnnlm_ver rnnlm-hs-0.1b \
  --stage 7 --N 100 --cmd "$decode_cmd" --inv-acwt 17 \
  0.4 data/lang_test_bd_fg data/local/rnnlm-hs.h400.voc40k data/test_eval92 \
  exp/tri3b/decode_bd_tgpr_eval92_fg $dir

dir=exp/tri3b/decode_bd_tgpr_eval92_fg_rnnlm-hs400_0.15
rm -rf $dir
cp -r exp/tri3b/decode_bd_tgpr_eval92_fg_rnnlm-hs400_0.3 $dir
steps/rnnlmrescore.sh --rnnlm_ver rnnlm-hs-0.1b \
  --stage 7 --N 100 --cmd "$decode_cmd" --inv-acwt 17 \
  0.15 data/lang_test_bd_fg data/local/rnnlm-hs.h400.voc40k data/test_eval92 \
  exp/tri3b/decode_bd_tgpr_eval92_fg $dir

steps/rnnlmrescore.sh --rnnlm_ver rnnlm-hs-0.1b \
  --N 10 --cmd "$decode_cmd" --inv-acwt 17 \
  0.3 data/lang_test_bd_fg data/local/rnnlm-hs.h400.voc40k data/test_eval92 \
  exp/tri3b/decode_bd_tgpr_eval92_fg exp/tri3b/decode_bd_tgpr_eval92_fg_rnnlm-hs400_0.3_N10 \
  || exit 1;

dir=exp/tri3b/decode_bd_tgpr_eval92_fg_rnnlm-hs400_0.4_N1000
rm -rf $dir
cp -r exp/tri3b/decode_bd_tgpr_eval92_fg_rnnlm-hs400_0.3_N1000 $dir
steps/rnnlmrescore.sh --rnnlm_ver rnnlm-hs-0.1b \
  --stage 7 --N 1000 --cmd "$decode_cmd" --inv-acwt 17 \
  0.4 data/lang_test_bd_fg data/local/rnnlm-hs.h400.voc40k data/test_eval92 \
  exp/tri3b/decode_bd_tgpr_eval92_fg $dir

dir=exp/tri3b/decode_bd_tgpr_eval92_fg_rnnlm-hs400_0.15_N1000
rm -rf $dir
cp -r exp/tri3b/decode_bd_tgpr_eval92_fg_rnnlm-hs400_0.3_N1000 $dir
steps/rnnlmrescore.sh --rnnlm_ver rnnlm-hs-0.1b \
  --stage 7 --N 1000 --cmd "$decode_cmd" --inv-acwt 17 \
  0.15 data/lang_test_bd_fg data/local/rnnlm-hs.h400.voc40k data/test_eval92 \
  exp/tri3b/decode_bd_tgpr_eval92_fg $dir

dir=exp/tri3b/decode_bd_tgpr_eval92_fg_rnnlm-hs400_0.5_N1000
rm -rf $dir
cp -r exp/tri3b/decode_bd_tgpr_eval92_fg_rnnlm-hs400_0.3_N1000 $dir
steps/rnnlmrescore.sh --rnnlm_ver rnnlm-hs-0.1b \
  --stage 7 --N 1000 --cmd "$decode_cmd" --inv-acwt 17 \
  0.5 data/lang_test_bd_fg data/local/rnnlm-hs.h400.voc40k data/test_eval92 \
  exp/tri3b/decode_bd_tgpr_eval92_fg $dir

dir=exp/tri3b/decode_bd_tgpr_eval92_fg_rnnlm-hs400_0.75_N1000
rm -rf $dir
cp -r exp/tri3b/decode_bd_tgpr_eval92_fg_rnnlm-hs400_0.3_N1000 $dir
steps/rnnlmrescore.sh --rnnlm_ver rnnlm-hs-0.1b \
  --stage 7 --N 1000 --cmd "$decode_cmd" --inv-acwt 17 \
  0.75 data/lang_test_bd_fg data/local/rnnlm-hs.h400.voc40k data/test_eval92 \
  exp/tri3b/decode_bd_tgpr_eval92_fg $dir
