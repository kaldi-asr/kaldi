#!/bin/bash

# This script demonstrates some commands that you could run after run_dnn.sh,
# that relate to conversion to the nnet2 model format.



steps/nnet2/convert_nnet1_to_nnet2.sh exp/dnn4b_pretrain-dbn_dnn exp/dnn4b_nnet2
cp exp/tri3b/splice_opts exp/tri3b/final.mat exp/dnn4b_nnet2
if [ -f exp/tri3b/cmvn_opts  ]; then
  cp exp/tri3b/cmvn_opts exp/dnn4b_nnet2
else
  echo -n >exp/dnn4b_nnet2/cmvn_opts
fi
 
steps/nnet2/decode.sh --nj 10 --cmd "$decode_cmd" --transform-dir exp/tri3b/decode \
   --config conf/decode.config exp/tri3b/graph data/test exp/dnn4b_nnet2/decode

# decoding results are essentially the same (any small difference is probably because
# decode.config != decode_dnn.config).
# %WER 1.58 [ 198 / 12533, 22 ins, 45 del, 131 sub ] exp/dnn4b_nnet2/decode/wer_3
# %WER 1.59 [ 199 / 12533, 23 ins, 45 del, 131 sub ] exp/dnn4b_pretrain-dbn_dnn/decode/wer_3


# This example puts the LDA in the model, but not the CMVN.
steps/nnet2/convert_lda_to_raw.sh exp/dnn4b_nnet2 exp/dnn4b_nnet2_raw
steps/nnet2/decode.sh --nj 10 --cmd "$decode_cmd" \
    --feat-type raw --config conf/decode.config exp/tri3b/graph data/test exp/dnn4b_nnet2_raw/decode

# This is worse because we're decoding without fMLLR.  It's OK, I just wanted to demonstrate
# the script, which I plan to use for systems without fMLLR.
# grep WER exp/dnn4b_nnet2_raw/decode/wer_* | utils/best_wer.sh 
# %WER 3.84 [ 481 / 12533, 44 ins, 136 del, 301 sub ] exp/dnn4b_nnet2_raw/decode/wer_7

matrix-sum scp:data/train/cmvn.scp global.cmvn
steps/nnet2/convert_lda_to_raw.sh --global-cmvn-stats global.cmvn exp/dnn4b_nnet2 exp/dnn4b_nnet2_raw_no_cmvn
rm global.cmvn
steps/nnet2/decode.sh --nj 10 --cmd "$decode_cmd" \
    --feat-type raw --config conf/decode.config exp/tri3b/graph data/test exp/dnn4b_nnet2_raw_no_cmvn/decode
# Even worse results, but this is expected due to the mismatch.
# grep WER exp/dnn4b_nnet2_raw_no_cmvn/decode/wer_* | utils/best_wer.sh 
# %WER 5.13 [ 643 / 12533, 82 ins, 144 del, 417 sub ] exp/dnn4b_nnet2_raw_no_cmvn/decode/wer_6


steps/online/nnet2/prepare_online_decoding.sh data/lang \
   exp/dnn4b_nnet2_raw_no_cmvn exp/dnn4b_nnet2_raw_no_cmvn_online

steps/online/nnet2/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 20 \
    exp/tri3b/graph data/test exp/dnn4b_nnet2_raw_no_cmvn_online/decode

# The following is decoding with the online-decoding code.
# grep WER exp/dnn4b_nnet2_raw_no_cmvn_online/decode/wer_* | utils/best_wer.sh 
# %WER 5.05 [ 633 / 12533, 79 ins, 141 del, 413 sub ] exp/dnn4b_nnet2_raw_no_cmvn_online/decode/wer_6
# It's slightly better than the offline decoding and I'm not sure why, as all the decoding
# parameters seem to be the same.  It may be some slight difference in how the lattices
# are determinized.
