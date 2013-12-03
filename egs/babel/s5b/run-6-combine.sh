#!/bin/bash


. conf/common_vars.sh
. ./lang.conf

set -e
set -o pipefail
set -u

# Wait till the main run.sh gets to the stage where's it's 
# finished aligning the tri5 model.

if [ ! -f exp/combine_2/decode_dev2h/.done ]; then
  for iter in 1 2 3 4; do
    local/score_combine.sh --cmd "queue.pl -l mem_free=2.0G,ram_free=1.0G" \
      data/dev2h data/lang exp/tri6_nnet/decode_dev2h exp/sgmm5_mmi_b0.1/decode_dev2h_fmllr_it$iter exp/combine_2/decode_dev2h_it$iter
    touch exp/combine_2/decode_dev2h/.done 
  done
fi

# This assumes the exp_BNF stuff is done..
if [ ! -f exp/combine_3/decode_dev2h/.done ]; then
  for iter in 1 2 3 4; do
    if [ ! -f exp_BNF/sgmm7_mmi_b0.1/decode_dev2h_fmllr_it$iter/.done ]; then
      echo "BNF decode in exp_BNF/sgmm7_mmi_b0.1/decode_dev2h_fmllr_it$iter is not done, skipping this step."
    fi
    local/score_combine.sh --cmd "queue.pl -l mem_free=2.0G,ram_free=1.0G" \
      data/dev2h data/lang exp_BNF/sgmm7_mmi_b0.1/decode_dev2h_fmllr_it$iter:10 \
      exp/sgmm5_mmi_b0.1/decode_dev2h_fmllr_it$iter exp/tri5_nnet/decode_dev2h exp/combine_3/decode_dev2h_it$iter
    touch exp_BNF/sgmm7_mmi_b0.1/decode_dev2h_fmllr_it$iter/.done
  done
fi

exit 0
