#!/bin/bash

# initialization PATH
. ./path.sh  || die "path.sh expected";
# initialization commands
. ./cmd.sh

#download all repo to build an ASR of Swahili
if [ ! -d "asr_swahili" ]; then
  #export from github
  svn co https://github.com/besacier/ALFFA_PUBLIC/trunk/ASR/SWAHILI/ asr_swahili || exit 1;
fi

[ ! -L "steps" ] && ln -s ../../wsj/s5/steps

[ ! -L "utils" ] && ln -s ../../wsj/s5/utils

[ ! -L "conf" ] && ln -s ../../wsj/s5/conf

# Data preparation
local/prepare_data.sh train test
local/prepare_dict.sh
##utils/prepare_lang.sh --position-dependent-phones false data/local/dict "<SIL>" data/local/lang data/lang
utils/prepare_lang.sh data/local/dict "<UNK>" data/local/lang data/lang
local/prepare_lm.sh

# Feature extraction
for x in train test; do
 steps/make_mfcc.sh --nj 20 --cmd "$train_cmd" data/$x exp/make_mfcc/$x mfcc
 steps/compute_cmvn_stats.sh data/$x exp/make_mfcc/$x mfcc
done

### Mono
# Training
steps/train_mono.sh --nj 20 --cmd "$train_cmd" data/train data/lang exp/system1/mono
# Graph compilation
utils/mkgraph.sh data/lang exp/system1/mono exp/system1/mono/graph

# Decoding
steps/decode.sh --nj 4 --cmd "$train_cmd" exp/system1/mono/graph  data/test exp/system1/mono/decode_test
echo -e "Mono training done.\n"


### Triphone
# Training
steps/align_si.sh --boost-silence 1.25 --nj 20 --cmd "$train_cmd" data/train data/lang exp/system1/mono exp/system1/mono_ali
steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" 3200 30000 data/train data/lang exp/system1/mono_ali exp/system1/tri1
# Graph compilation
utils/mkgraph.sh data/lang  exp/system1/tri1 exp/system1/tri1/graph
# Decoding
steps/decode.sh --nj 4 --cmd "$train_cmd" exp/system1/tri1/graph  data/test exp/system1/tri1/decode_test

## Triphones + delta delta
# Training
steps/align_si.sh --nj 20 --cmd "$train_cmd" data/train data/lang exp/system1/tri1 exp/system1/tri1_ali
steps/train_deltas.sh --cmd "$train_cmd" 3200 30000 data/train data/lang exp/system1/tri1_ali exp/system1/tri2a
# Graph compilation
utils/mkgraph.sh data/lang  exp/system1/tri2a exp/system1/tri2a/graph
# Decoding
steps/decode.sh --nj 4 --cmd "$train_cmd" exp/system1/tri2a/graph  data/test exp/system1/tri2a/decode_test
echo -e "Triphone training done.\n"


### Triphone + LDA and MLLT
# Training
steps/align_si.sh --nj 20 --cmd "$train_cmd" data/train data/lang exp/system1/tri2a exp/system1/tri2a_ali
steps/train_lda_mllt.sh --cmd "$train_cmd"  --splice-opts "--left-context=3 --right-context=3"   2000 20000 data/train data/lang  exp/system1/tri2a_ali exp/system1/tri2b
# Graph compilation
utils/mkgraph.sh data/lang  exp/system1/tri2b exp/system1/tri2b/graph
# Decoding
steps/decode.sh --nj 4 --cmd "$train_cmd" exp/system1/tri2b/graph  data/test exp/system1/tri2b/decode_test
echo -e "LDA+MLLT training done.\n"


### Triphone + LDA and MLLT + SAT and FMLLR
# Training
steps/align_si.sh --nj 20 --cmd "$train_cmd" --use-graphs true data/train data/lang exp/system1/tri2b exp/system1/tri2b_ali
steps/train_sat.sh --cmd "$train_cmd" 2000 20000 data/train data/lang exp/system1/tri2b_ali exp/system1/tri3b
# Graph compilation
utils/mkgraph.sh data/lang  exp/system1/tri3b exp/system1/tri3b/graph
# Decoding
steps/decode_fmllr.sh --nj 4 --cmd "$train_cmd" exp/system1/tri3b/graph  data/test exp/system1/tri3b/decode_test
#
steps/align_fmllr.sh --nj 20 --cmd "$train_cmd" data/train data/lang exp/system1/tri3b exp/system1/tri3b_ali
echo -e "SAT+FMLLR training done.\n"


### Triphone + LDA and MLLT + SAT and FMLLR + fMMI and MMI
# Training
steps/make_denlats.sh --nj 20 --cmd "$train_cmd" --sub-split 10 --transform-dir exp/system1/tri3b_ali data/train data/lang exp/system1/tri3b exp/system1/tri3b_denlats || exit 1;
steps/train_mmi.sh --cmd "$train_cmd" --boost 0.1 data/train data/lang exp/system1/tri3b_ali exp/system1/tri3b_denlats exp/system1/tri3b_mmi_b0.1  || exit 1;
# Decoding
steps/decode.sh --nj 4 --cmd "$train_cmd" --transform-dir exp/system1/tri3b/decode_test exp/system1/tri3b/graph data/test exp/system1/tri3b_mmi_b0.1/decode_test

## UBM for fMMI experiments
# Training
steps/train_diag_ubm.sh --silence-weight 0.5 --nj 20 --cmd "$train_cmd" 600 data/train data/lang exp/system1/tri3b_ali exp/system1/dubm3b

## fMMI+MMI
# Training
steps/train_mmi_fmmi.sh --cmd "$train_cmd" --boost 0.1 data/train data/lang exp/system1/tri3b_ali exp/system1/dubm3b exp/system1/tri3b_denlats exp/system1/tri3b_fmmi_a || exit 1;
# Decoding
for iter in 3 4 5 6 7 8; do
        steps/decode_fmmi.sh --nj 4 --cmd "$train_cmd" --iter $iter --transform-dir exp/system1/tri3b/decode_test  exp/system1/tri3b/graph data/test  exp/system1/tri3b_fmmi_a/decode_test_it$iter
done

## fMMI + mmi with indirect differential
# Training
steps/train_mmi_fmmi_indirect.sh --cmd "$train_cmd" --boost 0.1 data/train data/lang exp/system1/tri3b_ali exp/system1/dubm3b exp/system1/tri3b_denlats exp/system1/tri3b_fmmi_indirect || exit 1;
# Decoding
for iter in 3 4 5 6 7 8; do
 steps/decode_fmmi.sh --nj 4 --cmd "$train_cmd" --iter $iter --transform-dir exp/system1/tri3b/decode_test  exp/system1/tri3b/graph data/test exp/system1/tri3b_fmmi_indirect/decode_test_it$iter
done
echo -e "fMMI+MMI training done.\n"


### Triphone + LDA and MLLT + SGMM
## SGMM
# Training
steps/train_ubm.sh --cmd "$train_cmd" 500 data/train data/lang exp/system1/tri3b_ali exp/system1/ubm5b2 || exit 1;
steps/train_sgmm2.sh  --cmd "$train_cmd" 5000 12000 data/train data/lang exp/system1/tri3b_ali exp/system1/ubm5b2/final.ubm exp/system1/sgmm2_5b2 || exit 1;
# Graph compilation
utils/mkgraph.sh data/lang exp/system1/sgmm2_5b2 exp/system1/sgmm2_5b2/graph
# Decoding
steps/decode_sgmm2.sh --nj 4 --cmd "$train_cmd" --transform-dir exp/system1/tri3b/decode_test exp/system1/sgmm2_5b2/graph data/test exp/system1/sgmm2_5b2/decode_test
#
steps/align_sgmm2.sh --nj 20 --cmd "$train_cmd" --transform-dir exp/system1/tri3b_ali  --use-graphs true --use-gselect true data/train data/lang exp/system1/sgmm2_5b2 exp/system1/sgmm2_5b2_ali  || exit 1;

## Denlats
steps/make_denlats_sgmm2.sh --nj 20 --cmd "$train_cmd" --sub-split 10  --transform-dir exp/system1/tri3b_ali data/train data/lang exp/system1/sgmm2_5b2_ali exp/system1/sgmm2_5b2_denlats  || exit 1;

## SGMM+MMI
# Training
steps/train_mmi_sgmm2.sh --cmd "$train_cmd" --transform-dir exp/system1/tri3b_ali --boost 0.1 data/train data/lang exp/system1/sgmm2_5b2_ali exp/system1/sgmm2_5b2_denlats exp/system1/sgmm2_5b2_mmi_b0.1  || exit 1;

# Decoding
for iter in 1 2 3 4; do
        steps/decode_sgmm2_rescore.sh --cmd "$train_cmd" --iter $iter --transform-dir exp/system1/tri3b/decode_test data/lang data/test exp/system1/sgmm2_5b2/decode_test exp/system1/sgmm2_5b2_mmi_b0.1/decode_test_it$iter
done

# Training
steps/train_mmi_sgmm2.sh --cmd "$train_cmd" --transform-dir exp/tri3b_ali --boost 0.1 --drop-frames true data/train data/lang exp/system1/sgmm2_5b2_ali exp/system1/sgmm2_5b2_denlats exp/system1/sgmm2_5b2_mmi_b0.1_z
# Decoding
for iter in 1 2 3 4; do
        steps/decode_sgmm2_rescore.sh --cmd "$train_cmd" --iter $iter --transform-dir exp/system1/tri3b/decode_test data/lang data/test exp/system1/sgmm2_5b2/decode_test exp/system1/sgmm2_5b2_mmi_b0.1_z/decode_test_it$iter
done

## MBR
rm -r exp/system1/sgmm2_5b2_mmi_b0.1/decode_test_it3.mbr 2>/dev/null
cp -r exp/system1/sgmm2_5b2_mmi_b0.1/decode_test_it3{,.mbr}
local/score_mbr.sh data/test data/lang exp/system1/sgmm2_5b2_mmi_b0.1/decode_test_it3.mbr

## SGMM+MMI+fMMI
local/score_combine.sh data/test data/lang exp/system1/tri3b_fmmi_indirect/decode_test_it3  exp/system1/sgmm2_5b2_mmi_b0.1/decode_test_it3 exp/system1/combine_tri3b_fmmi_indirect_sgmm2_5b2_mmi_b0.1/decode_test_it8_3
echo -e "SGMM training done.\n"

#score
for x in exp/system1/*/decode*; do [ -d $x ] && grep WER $x/wer_* | utils/best_wer.sh; done
