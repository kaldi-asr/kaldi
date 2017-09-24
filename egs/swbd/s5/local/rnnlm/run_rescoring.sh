#!/bin/bash

n=50
ngram_order=4
rnndir=
id=rnn

. ./utils/parse_options.sh
. ./cmd.sh
. ./path.sh

set -e

#[ ! -f $rnndir/rnnlm ] && echo "Can't find RNNLM model" && exit 1;

LM=fsh_sw1_tg
rnndir=exp/rnnlm_lstm_h650_a

#ln -s final.raw $rnndir/rnnlm 2>/dev/null
touch $rnndir/unk.probs

for decode_set in eval2000; do
  dir=exp/chain/tdnn_lstm_1e_sp
  decode_dir=${dir}/decode_${decode_set}_$LM

  # N-best rescoring
#  steps/rnnlmrescore.sh \
#    --rnnlm-ver nnet3 \
#    --N $n --cmd "$decode_cmd --mem 16G" --inv-acwt 10 0.5 \
#    data/lang_$LM $rnndir \
#    data/$mic/$decode_set ${decode_dir} \
#    ${decode_dir}.$id.$n-best  &
#
#  continue

  # will implement later
  # Lattice rescoring
  steps/lmrescore_rnnlm_lat.sh \
    --cmd "$decode_cmd --mem 16G" \
    --rnnlm-ver kaldirnnlm  --weight 0.5 --max-ngram-order $ngram_order \
    data/lang_$LM $rnndir \
    data/${decode_set}_hires ${decode_dir} \
    ${decode_dir}.rnnlm.keli.nnet3rnnlm.lat.${ngram_order}gram

done

wait
