#!/bin/bash


. ./path.sh
. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.
nJobs=120
nDecodeJobs=40


galeData=GALE
mfccdir=mfcc

if [[ ! -d  exp/tri3b_ali ]]; then
  echo "exp/tri3b_ali lattices are required for alignmnet"
  exit 1
fi


## SGMM (subspace gaussian mixture model), excluding the "speaker-dependent weights"
steps/train_ubm.sh --cmd "$train_cmd" 700 \
 data/train data/lang exp/tri3b_ali exp/ubm5a || exit 1;

steps/train_sgmm2.sh --cmd "$train_cmd" 5000 20000 data/train data/lang exp/tri3b_ali \
  exp/ubm5a/final.ubm exp/sgmm_5a || exit 1;


utils/mkgraph.sh data/lang_test exp/sgmm_5a exp/sgmm_5a/graph
steps/decode_sgmm2.sh --nj $nDecodeJobs --cmd "$decode_cmd" \
 --config conf/decode.config --transform-dir \
 exp/tri3b/decode exp/sgmm_5a/graph data/test exp/sgmm_5a/decode


steps/align_sgmm2.sh --nj $nJobs --cmd "$train_cmd" --transform-dir exp/tri3b_ali \
  --use-graphs true --use-gselect true data/train data/lang exp/sgmm_5a exp/sgmm_5a_ali || exit 1;

## boosted MMI on SGMM
steps/make_denlats_sgmm2.sh --nj $nJobs --sub-split 30 --beam 9.0 --lattice-beam 6 \
  --cmd "$decode_cmd" --transform-dir \
  exp/tri3b_ali data/train data/lang exp/sgmm_5a_ali exp/sgmm_5a_denlats || exit 1;

steps/train_mmi_sgmm2.sh --cmd "$train_cmd" --num-iters 8 --transform-dir exp/tri3b_ali --boost 0.1 \
  data/train data/lang exp/sgmm_5a exp/sgmm_5a_denlats exp/sgmm_5a_mmi_b0.1

#decode SGMM MMI
utils/mkgraph.sh data/lang_test exp/sgmm_5a_mmi_b0.1 exp/sgmm_5a_mmi_b0.1/graph
steps/decode_sgmm2.sh --nj $nDecodeJobs --cmd "$decode_cmd" \
  --config conf/decode.config --transform-dir exp/tri3b/decode \
  exp/sgmm_5a_mmi_b0.1/graph data/test exp/sgmm_5a_mmi_b0.1/decode

for n in 1 2 3 4; do
  steps/decode_sgmm2_rescore.sh --cmd "$decode_cmd" --iter $n \
  --transform-dir exp/tri3b/decode data/lang_test data/test \
  exp/sgmm_5a_mmi_b0.1/decode exp/sgmm_5a_mmi_b0.1/decode$n
done

echo SGMM succedded
exit 0





