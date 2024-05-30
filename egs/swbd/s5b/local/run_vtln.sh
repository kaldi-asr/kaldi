#!/usr/bin/env bash

. ./cmd.sh
featdir=mfcc_vtln
num_leaves=3200
num_gauss=30000
logdet_scale=0.0

. parse_options.sh

# train linear vtln
steps/train_lvtln.sh --cmd "$train_cmd" \
  --logdet-scale $logdet_scale $num_leaves $num_gauss \
  data/train_30k_nodup data/lang exp/tri2 exp/tri2c || exit 1
mkdir -p data/train_30k_nodup_vtln
cp data/train_30k_nodup/* data/train_30k_nodup_vtln
cp exp/tri2c/final.warp data/train_30k_nodup_vtln/spk2warp || exit 1
steps/make_mfcc.sh --compress true --nj 20 --cmd "$train_cmd" data/train_30k_nodup_vtln exp/make_mfcc/train_30k_nodup_vtln ${featdir} || exit 1
steps/compute_cmvn_stats.sh data/train_30k_nodup_vtln exp/make_mfcc/train_30k_nodup_vtln ${featdir} || exit 1
utils/fix_data_dir.sh data/train_30k_nodup_vtln || exit 1 # remove segments with problems

utils/mkgraph.sh data/lang_sw1_tg exp/tri2c exp/tri2c/graph_sw1_tg || exit 1
steps/decode_lvtln.sh --config conf/decode.config --nj 30 --cmd "$decode_cmd" --logdet-scale $logdet_scale \
  exp/tri2c/graph_sw1_tg data/eval2000 exp/tri2c/decode_eval2000_sw1_tg || exit 1

mkdir -p data/eval2000_vtln
cp data/eval2000/* data/eval2000_vtln
cp exp/tri2c/decode_eval2000_sw1_tg/final.warp data/eval2000_vtln/spk2warp || exit 1
steps/make_mfcc.sh --cmd "$train_cmd" --nj 10 data/eval2000_vtln exp/make_mfcc/eval2000_vtln ${featdir} || exit 1
steps/compute_cmvn_stats.sh data/eval2000_vtln exp/make_mfcc/eval2000_vtln ${featdir} || exit 1
utils/fix_data_dir.sh data/eval2000_vtln  || exit 1 # remove segments with problems

steps/align_lvtln.sh --nj 30 --cmd "$train_cmd" --logdet-scale $logdet_scale \
  data/train_100k_nodup data/lang exp/tri2c exp/tri2c_ali_100k_nodup || exit 1
mkdir data/train_100k_nodup_vtln
cp data/train_100k_nodup/* data/train_100k_nodup_vtln
cp exp/tri2c_ali_100k_nodup/final.warp data/train_100k_nodup_vtln/spk2warp || exit 1

steps/train_lda_mllt.sh --cmd "$train_cmd" \
  5500 90000 data/train_100k_nodup_vtln data/lang exp/tri2c_ali_100k_nodup exp/tri3c  || exit 1

for lm_suffix in tg fsh_tgpr; do
  (
  graph_dir=exp/tri3c/graph_sw1_${lm_suffix}
  $train_cmd $graph_dir/mkgraph.log \
    utils/mkgraph.sh data/lang_sw1_${lm_suffix} exp/tri3c $graph_dir || exit 1
  steps/decode.sh --nj 30 --cmd "$decode_cmd" --config conf/decode.config \
    $graph_dir data/eval2000_vtln exp/tri3c/decode_eval2000_sw1_${lm_suffix} || exit 1
  ) &
done

# Train tri4a, which is LDA+MLLT+SAT, on 100k_nodup data.
steps/align_fmllr.sh --nj 30 --cmd "$train_cmd" \
  data/train_100k_nodup_vtln data/lang exp/tri3c exp/tri3c_ali_100k_nodup 

steps/train_sat.sh  --cmd "$train_cmd" \
  5500 90000 data/train_100k_nodup_vtln data/lang exp/tri3c_ali_100k_nodup \
   exp/tri4c 

for lm_suffix in tg fsh_tgpr; do
  (
    graph_dir=exp/tri4c/graph_sw1_${lm_suffix}
    $train_cmd $graph_dir/mkgraph.log \
      utils/mkgraph.sh data/lang_sw1_${lm_suffix} exp/tri4c $graph_dir
    steps/decode_fmllr.sh --nj 30 --cmd "$decode_cmd" --config conf/decode.config \
      $graph_dir data/eval2000_vtln exp/tri4c/decode_eval2000_sw1_${lm_suffix}
  ) &
done

# Baseline without VTLN
# %WER 24.5 | 1831 21395 | 78.0 15.1 6.8 2.5 24.5 62.5 | exp/tri4a/decode_eval2000_sw1_fsh_tgpr/score_15/eval2000.ctm.swbd.filt.sys

# With VTLN
# %WER 24.1 | 1831 21395 | 78.3 15.0 6.7 2.5 24.1 60.8 | exp/tri4c/decode_eval2000_sw1_fsh_tgpr/score_15/eval2000_vtln.ctm.swbd.filt.sys

