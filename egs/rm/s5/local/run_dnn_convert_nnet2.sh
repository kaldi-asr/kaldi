#!/bin/bash

# This script demonstrates some commands that you could run after run_dnn.sh,
# that relate to conversion to the nnet2 model format.



steps/nnet2/convert_nnet1_to_nnet2.sh exp/dnn4b_pretrain-dbn_dnn exp/dnn4b_nnet2
cp exp/tri3b/splice_opts exp/tri3b/cmvn_opts exp/tri3b/final.mat exp/dnn4b_nnet2/
 
steps/nnet2/decode.sh --nj 10 --cmd "$decode_cmd" --transform-dir exp/tri3b/decode \
   --config conf/decode.config exp/tri3b/graph data/test exp/dnn4b_nnet2/decode

# decoding results are essentially the same (any small difference is probably because
# decode.config != decode_dnn.config).
# %WER 1.58 [ 198 / 12533, 22 ins, 45 del, 131 sub ] exp/dnn4b_nnet2/decode/wer_3
# %WER 1.59 [ 199 / 12533, 23 ins, 45 del, 131 sub ] exp/dnn4b_pretrain-dbn_dnn/decode/wer_3


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
# %WER 5.31 [ 666 / 12533, 76 ins, 163 del, 427 sub ] exp/dnn4b_nnet2_raw_no_cmvn/decode/wer_7


