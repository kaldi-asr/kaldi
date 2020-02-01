#!/usr/bin/env bash

. ./cmd.sh
. ./path.sh

# This script demonstrates how you can train rnnlms, and how you can use them to
# rescore the n-best lists, or lattices.
# Be careful: appending things like "--mem 16G" to $decode_cmd won't always
# work, it depends what $decode_cmd is.

# Trains Tomas Mikolov's version, which takes roughly 5 days with the following
# parameter setting. We start from the dictionary directory without silence
# probabilities (with suffix "_nosp").
rm data/local/rnnlm.h300.voc40k/.error 2>/dev/null
local/wsj_train_rnnlms.sh --dict-suffix "_nosp" \
  --cmd "$decode_cmd --mem 16G" \
  --hidden 300 --nwords 40000 --class 400 \
  --direct 2000 data/local/rnnlm.h300.voc40k \
  || touch data/local/rnnlm.h300.voc40k/.error &

# Trains Yandex's version, which takes roughly 10 hours with the following
# parameter setting. We start from the dictionary directory without silence
# probabilities (with suffix "_nosp").
num_threads_rnnlm=8
rm data/local/rnnlm-hs.nce20.h400.voc40k/.error 2>/dev/null
local/wsj_train_rnnlms.sh --dict-suffix "_nosp" \
  --rnnlm_ver faster-rnnlm --threads $num_threads_rnnlm \
  --cmd "$decode_cmd --mem 8G --num-threads $num_threads_rnnlm" \
  --bptt 4 --bptt-block 10 --hidden 400 --nwords 40000 --direct 2000 \
  --rnnlm-options "-direct-order 4 -nce 20" \
  data/local/rnnlm-hs.nce20.h400.voc40k \
  || touch data/local/rnnlm-hs.nce20.h400.voc40k/.error &

wait;

# Rescoring. We demonstrate results on the TDNN models. Make sure you have
# finished running the following scripts:
#   local/online/run_nnet2.sh
#   local/online/run_nnet2_baseline.sh
#   local/online/run_nnet2_discriminative.sh
for lm_suffix in tgpr bd_tgpr; do
  graph_dir=exp/tri4b/graph_${lm_suffix}
  for year in eval92 dev93; do
    decode_dir=exp/nnet2_online/nnet_ms_a_online/decode_${lm_suffix}_${year}

    # N-best rescoring with Tomas Mikolov's version.
    steps/rnnlmrescore.sh \
      --N 1000 --cmd "$decode_cmd --mem 16G" --inv-acwt 10 0.75 \
      data/lang_test_${lm_suffix} data/local/rnnlm.h300.voc40k \
      data/test_${year} ${decode_dir} \
      ${decode_dir}_rnnlm.h300.voc40k || exit 1;

    # Lattice rescoring with Tomas Mikolov's version.
    steps/lmrescore_rnnlm_lat.sh \
      --weight 0.75 --cmd "$decode_cmd --mem 16G" --max-ngram-order 5 \
      data/lang_test_${lm_suffix} data/local/rnnlm.h300.voc40k \
      data/test_${year} ${decode_dir} \
      ${decode_dir}_rnnlm.h300.voc40k_lat || exit 1;

    # N-best rescoring with Yandex's version.
    steps/rnnlmrescore.sh --rnnlm_ver faster-rnnlm \
      --N 1000 --cmd "$decode_cmd --mem 8G" --inv-acwt 10 0.75 \
      data/lang_test_${lm_suffix} data/local/rnnlm-hs.nce20.h400.voc40k \
      data/test_${year} ${decode_dir} \
      ${decode_dir}_rnnlm-hs.nce20.h400.voc40k || exit 1;
  done
done
