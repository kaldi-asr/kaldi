#!/bin/bash 


. ./path.sh
. ./cmd.sh ## You'll want to change cmd.sh to something that will work on your system.
           ## This relates to the queue.
nJobs=120
nDecodeJobs=40

galeData=GALE
mfccdir=mfcc


 
if [[ ! -e  exp/tri2b/final.mdl ]]; then 
  echo "exp/tri2b/final.mdl is required for SGMM"
  exit 1 
fi


#  Do MMI on top of LDA+MLLT.
steps/make_denlats.sh --nj $nJobs --cmd "$train_cmd" \
 data/train data/lang exp/tri2b exp/tri2b_denlats || exit 1;
steps/train_mmi.sh --cmd "$train_cmd" data/train data/lang exp/tri2b_ali \
 exp/tri2b_denlats exp/tri2b_mmi || exit 1;


  steps/decode.sh  --iter 4 --nj $nJobs --cmd "$decode_cmd"  \
  exp/tri2b/graph data/test exp/tri2b_mmi/decode_it4 && \
  steps/decode.sh  --iter 3 --nj $nJobs --cmd "$decode_cmd" \
  exp/tri2b/graph data/test exp/tri2b_mmi/decode_it3 

steps/train_mmi.sh --cmd "$train_cmd" --boost 0.05 data/train data/lang exp/tri2b_ali \
exp/tri2b_denlats exp/tri2b_mmi_b0.05 || exit 1;

  steps/decode.sh  --iter 4 --nj $nJobs --cmd "$decode_cmd" \
  exp/tri2b/graph data/test exp/tri2b_mmi_b0.05/decode_it4 && \
  steps/decode.sh  --iter 3 --nj $nJobs --cmd "$decode_cmd" \
  exp/tri2b/graph data/test exp/tri2b_mmi_b0.05/decode_it3 
s and 
# Do MPE.
steps/train_mpe.sh --cmd "$train_cmd" data/train data/lang exp/tri2b_ali exp/tri2b_denlats exp/tri2b_mpe || exit 1;

  steps/decode.sh  --iter 4 --nj $nDecodeJobs --cmd "$decode_cmd" \
  exp/tri2b/graph data/test exp/tri2b_mpe/decode_it4 && \
  steps/decode.sh  --iter 3 --nj $nDecodeJobs --cmd "$decode_cmd" \
  exp/tri2b/graph data/test exp/tri2b_mpe/decode_it3


  
echo training mmi mpe succedded
exit 0





