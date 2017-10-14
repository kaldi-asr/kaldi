#!/bin/bash

ngram_order=4
rnndir=exp/rnnlm_lstm_e

. ./utils/parse_options.sh
. ./cmd.sh
. ./path.sh

set -e

LM=fsh_sw1_tg

./local/rnnlm/tuning/run_lstm_e.sh

for decode_set in eval2000; do
  dir=exp/chain/tdnn_lstm_1e_sp
  decode_dir=${dir}/decode_${decode_set}_$LM

  # Lattice rescoring
  rnnlm/lmrescore_rnnlm_lat.sh \
    --cmd "$decode_cmd --mem 16G -l hostname=b*" \
    --weight 0.5 --max-ngram-order $ngram_order \
    data/lang_$LM $rnndir \
    data/${decode_set}_hires ${decode_dir} \
    ${decode_dir}.kaldirnnlm.lat.${ngram_order}gram

done
