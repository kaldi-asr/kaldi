#!/bin/bash

mic=sdm1

. ./utils/parse_options.sh
. ./cmd.sh
. ./path.sh

#local/train_cued_rnnlms.sh --train-text data/$mic/train/text data/$mic/cued_rnn

final_lm=ami_fsh.o3g.kn
LM=$final_lm.pr1-7

graph_dir=exp/$mic/tri4a/graph_${LM}

for decode_set in dev eval; do
  dir=exp/$mic/nnet3/tdnn_sp/
  decode_dir=${dir}/decode_${decode_set}

  # Lattice rescoring
  steps/lmrescore_rnnlm_lat.sh \
    --rnnlm-ver cuedrnnlm \
    --layer-string "10002 200 10002" \
    --weight 0.5 --cmd "$decode_cmd --mem 16G" --max-ngram-order 3 \
    data/$mic/cued_rnn data/lang_$LM \
    data/$mic/$decode_set ${decode_dir} \
    ${decode_dir}.rnnlm.cued.lat || exit 1;

  continue
  # N-best rescoring
  steps/rnnlmrescore.sh \
    --rnnlm-ver cuedrnnlm \
    --N 50 --cmd "$decode_cmd --mem 16G" --inv-acwt 10 0.5 \
    data/lang_$LM data/$mic/cued_rnn \
    data/$mic/$decode_set ${decode_dir} \
    ${decode_dir}.rnnlm.cued.50-best

  continue
done

