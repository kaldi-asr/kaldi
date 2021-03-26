#!/usr/bin/env bash

. ./cmd.sh
featdir=mfcc

# train linear vtln
set -e

steps/train_lvtln.sh --cmd "$train_cmd" 1800 9000 \
   data/train data/lang exp/tri2b exp/tri3e

mkdir -p data/train_vtln
cp data/train/* data/train_vtln || true
cp exp/tri3e/final.warp data/train_vtln/spk2warp
steps/make_mfcc.sh --nj 8 --cmd "run.pl" data/train_vtln exp/make_mfcc/train_vtln $featdir  
steps/compute_cmvn_stats.sh data/train_vtln exp/make_mfcc/train_vtln $featdir  
 utils/mkgraph.sh data/lang exp/tri3e exp/tri3e/graph
steps/decode_lvtln.sh --config conf/decode.config --nj 20 --cmd "$decode_cmd" \
  exp/tri3e/graph data/test exp/tri3e/decode

mkdir -p data/test_vtln
cp data/test/* data/test_vtln || true
cp exp/tri3e/decode/final.warp data/test_vtln/spk2warp
steps/make_mfcc.sh --nj 8 --cmd "run.pl" data/test_vtln exp/make_mfcc/test_vtln $featdir  
steps/compute_cmvn_stats.sh data/test_vtln exp/make_mfcc/test_vtln $featdir  

(
 steps/train_lda_mllt.sh --cmd "$train_cmd" \
   --splice-opts "--left-context=3 --right-context=3" \
  1800 9000 data/train_vtln data/lang exp/tri3e exp/tri4e
 utils/mkgraph.sh data/lang exp/tri4e exp/tri4e/graph

 steps/decode.sh --config conf/decode.config --nj 20 --cmd "$decode_cmd" \
    exp/tri4e/graph data/test_vtln exp/tri4e/decode

 steps/train_sat.sh 1800 9000 data/train_vtln data/lang exp/tri4e exp/tri5e
 utils/mkgraph.sh data/lang exp/tri5e exp/tri5e/graph 
 steps/decode_fmllr.sh --config conf/decode.config --nj 20 --cmd "$decode_cmd" \
   exp/tri5e/graph data/test_vtln exp/tri5e/decode 

 utils/mkgraph.sh data/lang_ug exp/tri5e exp/tri5e/graph_ug
 steps/decode_fmllr.sh --config conf/decode.config --nj 20 --cmd "$decode_cmd" \
   exp/tri5e/graph_ug data/test_vtln exp/tri5e/decode_ug
)


# Below is the results we got from running this script.  5e is  with
# VTLN, and 3b is the baseline.  VTLN helps at the speaker independent
# pass, but there is essentially no difference after adaptation (except
# for a small improvement on the _ug decode, from 10.36 to 10.14).
#
# a04:s5: for x in exp/tri5e/decode*; do grep WER $x/wer_* | utils/best_wer.sh; done
# %WER 1.95 [ 245 / 12533, 31 ins, 60 del, 154 sub ] exp/tri5e/decode/wer_6
# %WER 2.46 [ 308 / 12533, 38 ins, 67 del, 203 sub ] exp/tri5e/decode.si/wer_6
# %WER 10.14 [ 1271 / 12533, 120 ins, 204 del, 947 sub ] exp/tri5e/decode_ug/wer_13
# %WER 11.59 [ 1453 / 12533, 132 ins, 253 del, 1068 sub ] exp/tri5e/decode_ug.si/wer_13
# a04:s5: for x in exp/tri3b/decode*; do grep WER $x/wer_* | utils/best_wer.sh; done
# %WER 1.95 [ 245 / 12533, 21 ins, 63 del, 161 sub ] exp/tri3b/decode/wer_7
# %WER 3.13 [ 392 / 12533, 59 ins, 64 del, 269 sub ] exp/tri3b/decode.si/wer_3
# %WER 10.36 [ 1298 / 12533, 147 ins, 192 del, 959 sub ] exp/tri3b/decode_ug/wer_12
# %WER 13.48 [ 1689 / 12533, 159 ins, 277 del, 1253 sub ] exp/tri3b/decode_ug.si/wer_13
# a04:s5: 
