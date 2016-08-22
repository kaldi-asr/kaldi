#!/bin/bash

mic=sdm1

. ./utils/parse_options.sh
. ./cmd.sh
. ./path.sh

local/train_rnnlms.sh --train-text data/$mic/train/text data/$mic/mik_rnn

final_lm=ami_fsh.o3g.kn
LM=$final_lm.pr1-7

graph_dir=exp/$mic/tri4a/graph_${LM}


for decode_set in eval dev; do
  dir=exp/$mic/nnet3/tdnn_sp/
  decode_dir=${dir}/decode_${decode_set}

  # N-best rescoring with Tomas Mikolov's version.
(  steps/rnnlmrescore.sh \
    --rnnlm-ver rnnlm-0.3e \
    --N 50 --cmd "$decode_cmd --mem 16G" --inv-acwt 10 0.5 \
    data/lang_$LM data/$mic/mik_rnn \
    data/$mic/$decode_set ${decode_dir} \
    ${decode_dir}.rnnlm.mik.50-best || exit 1 ) &

  # Lattice rescoring with Tomas Mikolov's version.
(  steps/lmrescore_rnnlm_lat.sh \
    --weight 0.5 --cmd "$decode_cmd --mem 16G" --max-ngram-order 5 \
    data/lang_$LM data/$mic/mik_rnn \
    data/$mic/$decode_set ${decode_dir} \
    ${decode_dir}.rnnlm.mik.lat || exit 1;) &
done

wait

