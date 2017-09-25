#!/bin/bash

n=50
ngram_order=4
rnndir=
id=rnn

. ./utils/parse_options.sh
. ./cmd.sh
. ./path.sh

set -e

LM=fsh_sw1_tg
rnndir=exp/rnnlm_lstm_d


for decode_set in eval2000; do
  dir=exp/chain/tdnn_lstm_1e_sp
  decode_dir=${dir}/decode_${decode_set}_$LM

  # Lattice rescoring
  rnnlm/lmrescore_rnnlm_lat.sh \
    --cmd "$decode_cmd --mem 16G" \
    --rnnlm-ver kaldirnnlm  --weight 0.5 --max-ngram-order $ngram_order \
    data/lang_$LM $rnndir \
    data/${decode_set}_hires ${decode_dir} \
    ${decode_dir}.nnet3rnnlm.lat.${ngram_order}gram

done

wait
