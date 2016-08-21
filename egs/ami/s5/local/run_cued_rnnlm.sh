#!/bin/bash

mic=sdm1
crit=vr
n=50
ngram_order=4

. ./utils/parse_options.sh
. ./cmd.sh
. ./path.sh

set -e

local/train_cued_rnnlms.sh --crit $crit --train-text data/$mic/train/text data/$mic/cued_rnn_$crit

final_lm=ami_fsh.o3g.kn
LM=$final_lm.pr1-7

for decode_set in dev eval; do
  dir=exp/$mic/nnet3/tdnn_sp/
  decode_dir=${dir}/decode_${decode_set}

  # N-best rescoring
  steps/rnnlmrescore.sh \
    --rnnlm-ver cuedrnnlm \
    --N $n --cmd "$decode_cmd --mem 16G" --inv-acwt 10 0.5 \
    data/lang_$LM data/$mic/cued_rnn_$crit \
    data/$mic/$decode_set ${decode_dir} \
    ${decode_dir}.rnnlm.$crit.cued.$n-best 

  # Lattice rescoring
  steps/lmrescore_rnnlm_lat.sh \
    --cmd "$decode_cmd --mem 16G" \
    --rnnlm-ver cuedrnnlm  --weight 0.5 --max-ngram-order $ngram_order \
    data/lang_$LM data/$mic/cued_rnn_$crit \
    data/$mic/${decode_set}_hires ${decode_dir} \
    ${decode_dir}.rnnlm.$crit.cued.lat.${ngram_order}gram

done
