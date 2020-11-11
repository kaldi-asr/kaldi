#!/usr/bin/env bash

for test in dev93 eval92; do

  steps/lmrescore.sh --cmd "$decode_cmd" data/lang_test_bd_tgpr data/lang_test_bd_fg \
    data/test_${test} exp/sgmm5b_mmi_b0.1/decode_bd_tgpr_${test}_it4 exp/sgmm5b_mmi_b0.1/decode_bd_fg_${test}_it4 || exit 1;


# Note: for N-best-list generation, choosing the acoustic scale (12) that gave
# the best WER on this test set.  Ideally we should do this on a dev set.

 # This step interpolates a small RNNLM (with weight 0.25) with the 4-gram LM.
  steps/rnnlmrescore.sh \
    --N 100 --cmd "$decode_cmd" --inv-acwt 12 \
    0.25 data/lang_test_bd_fg data/local/rnnlm.h30.voc10k data/test_${test} \
    exp/sgmm5b_mmi_b0.1/decode_bd_fg_${test}_it4 exp/sgmm5b_mmi_b0.1/decode_bd_fg_${test}_it4_rnnlm30_0.25  \
    || exit 1;

  steps/rnnlmrescore.sh \
    --N 100 --cmd "$decode_cmd" --inv-acwt 12 \
    0.5 data/lang_test_bd_fg data/local/rnnlm.h100.voc20k data/test_${test} \
    exp/sgmm5b_mmi_b0.1/decode_bd_fg_${test}_it4 exp/sgmm5b_mmi_b0.1/decode_bd_fg_${test}_it4_rnnlm100_0.5 \
    || exit 1;

  steps/rnnlmrescore.sh \
    --N 100 --cmd "$decode_cmd" --inv-acwt 12 \
    0.5 data/lang_test_bd_fg data/local/rnnlm.h200.voc30k data/test_${test} \
    exp/sgmm5b_mmi_b0.1/decode_bd_fg_${test}_it4 exp/sgmm5b_mmi_b0.1/decode_bd_fg_${test}_it4_rnnlm200_0.5 \
    || exit 1;

  steps/rnnlmrescore.sh \
    --N 100 --cmd "$decode_cmd" --inv-acwt 12 \
    0.5 data/lang_test_bd_fg data/local/rnnlm.h300.voc40k data/test_${test} \
    exp/sgmm5b_mmi_b0.1/decode_bd_fg_${test}_it4 exp/sgmm5b_mmi_b0.1/decode_bd_fg_${test}_it4_rnnlm300_0.5 \
    || exit 1;

  steps/rnnlmrescore.sh \
    --N 100 --cmd "$decode_cmd" --inv-acwt 12 \
    0.75 data/lang_test_bd_fg data/local/rnnlm.h300.voc40k data/test_${test} \
    exp/sgmm5b_mmi_b0.1/decode_bd_fg_${test}_it4 exp/sgmm5b_mmi_b0.1/decode_bd_fg_${test}_it4_rnnlm300_0.75 \
    || exit 1;
done
