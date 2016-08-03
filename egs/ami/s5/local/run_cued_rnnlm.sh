#!/bin/bash

mic=sdm1

. ./utils/parse_options.sh
. ./cmd.sh
. ./path.sh

set -e

#local/train_cued_rnnlms.sh --crit vr --train-text data/$mic/train/text data/$mic/cued_rnn_vr

final_lm=ami_fsh.o3g.kn
LM=$final_lm.pr1-7

for decode_set in dev eval; do
  dir=exp/$mic/nnet3/tdnn_sp/
  decode_dir=${dir}/decode_${decode_set}

  false && (
  # N-best rescoring
  steps/rnnlmrescore.sh \
    --rnnlm-ver cuedrnnlm \
    --N 50 --cmd "$decode_cmd --mem 16G" --inv-acwt 10 0.5 \
    data/lang_$LM data/$mic/cued_rnn_vr \
    data/$mic/$decode_set ${decode_dir} \
    ${decode_dir}.rnnlm.vr.cued.50-best

    ) 

  (
  # Lattice rescoring
  steps/lmrescore_rnnlm_lat.sh \
    --rnnlm-ver cuedrnnlm \
    --layer-string "10002 200 10002" \
    --weight 0.5 --cmd "$decode_cmd --mem 16G" --max-ngram-order 4 \
    data/lang_$LM data/$mic/cued_rnn_vr \
    data/$mic/${decode_set}_hires ${decode_dir} \
    ${decode_dir}.rnnlm.vr.cued.lat.4gram || exit 1;
  ) &

  (
  # Lattice rescoring
  steps/lmrescore_rnnlm_lat.sh \
    --rnnlm-ver cuedrnnlm \
    --layer-string "10002 200 10002" \
    --weight 0.5 --cmd "$decode_cmd --mem 16G" --max-ngram-order 5 \
    data/lang_$LM data/$mic/cued_rnn_vr \
    data/$mic/${decode_set}_hires ${decode_dir} \
    ${decode_dir}.rnnlm.vr.cued.lat.5gram || exit 1;
  ) &

  continue
done

wait

