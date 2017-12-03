#!/bin/bash

. ./utils/parse_options.sh
. ./cmd.sh
. ./path.sh

set -e

rnndir=exp/rnnlm_lstm_e

LM=fsh_sw1_tg

weight=0.8 # the weight of the RNNLM in rescoring

for decode_set in eval2000; do
  dir=exp/chain/tdnn_lstm_1e_sp
  decode_dir=${dir}/decode_${decode_set}_$LM

  # Lattice rescoring
  rnnlm/lmrescore_pruned.sh \
    --cmd "$decode_cmd" \
    --weight $weight --max-ngram-order $ngram_order \
    data/lang_$LM $rnndir \
    data/${decode_set}_hires ${decode_dir} \
    ${decode_dir}_rnnlm_${ngram_order}gram

#  rnnlm/lmrescore.sh \
#    --cmd "$decode_cmd" \
#    --weight $weight --max-ngram-order $ngram_order \
#    data/lang_$LM $rnndir \
#    data/${decode_set}_hires ${decode_dir} \
#    ${decode_dir}_rnnlm_${ngram_order}gram_unpruned
done
