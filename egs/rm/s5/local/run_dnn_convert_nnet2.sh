#!/bin/bash

# This script demonstrates some commands that you could run after run_dnn.sh,
# that relate to conversion to the nnet2 model format.



steps/nnet2/convert_nnet1_to_nnet2.sh exp/dnn4b_pretrain-dbn_dnn exp/dnn4b_nnet2
 
steps/nnet2/decode.sh --nj 10 --cmd "$decode_cmd" --transform-dir exp/tri3b/decode \
   --config conf/decode.config exp/tri3b/graph data/test exp/dnn4b_nnet2/decode
steps/nnet2/decode.sh --nj 10 --cmd "$decode_cmd" --transform-dir exp/tri3b/decode \
   --config conf/decode.config exp/tri3b/graph_ug data/test exp/dnn4b_nnet2/decode_ug

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


( # We demonstrate doing further training on top of a model initially
  # trained by Karel's tools.
  nnet-am-switch-preconditioning exp/dnn4b_nnet2/final.mdl - | \
    nnet-am-copy --learning-rate=0.001 - exp/dnn4b_nnet2/final.mdl.mod

  mkdir -p exp/dnn4b_nnet2_retrain

  steps/nnet2/get_egs.sh --samples-per-iter 200000 \
    --num-jobs-nnet 4 --splice-width 5 --cmd "$train_cmd" \
    data/train data/lang exp/tri3b_ali \
    exp/dnn4b_nnet2_retrain

 # options here are for GPU use.
  steps/nnet2/train_more.sh --learning-rate-factor 0.1 --cmd "$train_cmd" \
    --parallel-opts "-l gpu=1" --num-threads 1  --minibatch-size 512 \
    exp/dnn4b_nnet2/final.mdl.mod exp/dnn4b_nnet2_retrain/egs exp/dnn4b_nnet2_retrain

  steps/nnet2/decode.sh --nj 30 --cmd "$decode_cmd" --transform-dir exp/tri3b/decode \
    --config conf/decode.config exp/tri3b/graph data/test exp/dnn4b_nnet2_retrain/decode
  steps/nnet2/decode.sh --nj 30 --cmd "$decode_cmd" --transform-dir exp/tri3b/decode \
    --config conf/decode.config exp/tri3b/graph_ug data/test exp/dnn4b_nnet2_retrain/decode_ug
 #Results for this experiment:
 #for x in exp/dnn4b_nnet2_retrain/decode*; do grep WER $x/wer_* | utils/best_wer.sh ; done
 #%WER 1.58 [ 198 / 12533, 29 ins, 38 del, 131 sub ] exp/dnn4b_nnet2_retrain/decode/wer_3
 #%WER 7.60 [ 953 / 12533, 56 ins, 168 del, 729 sub ] exp/dnn4b_nnet2_retrain/decode_ug/wer_10

 # vs. the following baseline (our experiment got 0.2% abs. improvement on unigram only).
 #for x in exp/dnn4b_nnet2/decode*; do grep WER $x/wer_* | utils/best_wer.sh ; done
 # %WER 1.58 [ 198 / 12533, 22 ins, 45 del, 131 sub ] exp/dnn4b_nnet2/decode/wer_3
 #%WER 7.80 [ 977 / 12533, 83 ins, 151 del, 743 sub ] exp/dnn4b_nnet2/decode_ug/wer_6

)

(
  # We demonstrate doing further training on top of a DBN trained
  # generatively by Karel's tools.
  mkdir -p exp/dnn4b_nnet2_dbn_in
  for f in final.mdl final.feature_transform ali_train_pdf.counts; do
    cp exp/dnn4b_pretrain-dbn_dnn/$f exp/dnn4b_nnet2_dbn_in/
  done
  cp exp/dnn4b_pretrain-dbn/6.dbn exp/dnn4b_nnet2_dbn_in/final.dbn
  steps/nnet2/convert_nnet1_to_nnet2.sh exp/dnn4b_nnet2_dbn_in exp/dnn4b_nnet2_dbn
  cp exp/tri3b/splice_opts exp/tri3b/cmvn_opts exp/tri3b/final.mat exp/tri3b/tree exp/dnn4b_nnet2_dbn/


  nnet-am-switch-preconditioning exp/dnn4b_nnet2_dbn/final.mdl - | \
    nnet-am-copy --learning-rate=0.01 - exp/dnn4b_nnet2_dbn/final.mdl.mod

  steps/nnet2/get_egs.sh --samples-per-iter 200000 \
     --num-jobs-nnet 4 --splice-width 5 --cmd "$train_cmd" \
     data/train data/lang exp/tri3b_ali \
      exp/dnn4b_nnet2_dbn_retrain

  steps/nnet2/train_more.sh --learning-rate-factor 0.1 --cmd "$train_cmd" \
    --parallel-opts "-l gpu=1" --num-threads 1  --minibatch-size 512 \
    exp/dnn4b_nnet2_dbn/final.mdl.mod exp/dnn4b_nnet2_dbn_retrain/egs exp/dnn4b_nnet2_dbn_retrain


  steps/nnet2/decode.sh --nj 30 --cmd "$decode_cmd" --transform-dir exp/tri3b/decode \
    --config conf/decode.config exp/tri3b/graph data/test exp/dnn4b_nnet2_dbn_retrain/decode &
  steps/nnet2/decode.sh --nj 30 --cmd "$decode_cmd" --transform-dir exp/tri3b/decode \
    --config conf/decode.config exp/tri3b/graph_ug data/test exp/dnn4b_nnet2_dbn_retrain/decode_ug &

 # Here are the results (and note that we never tuned this at all, it was our first guess
 # at what might be good parameters).
 #for x in exp/dnn4b_nnet2_dbn_retrain/decode*; do grep WER $x/wer_* | utils/best_wer.sh ; done
 #%WER 1.68 [ 210 / 12533, 36 ins, 43 del, 131 sub ] exp/dnn4b_nnet2_dbn_retrain/decode/wer_3
 #%WER 7.86 [ 985 / 12533, 72 ins, 172 del, 741 sub ] exp/dnn4b_nnet2_dbn_retrain/decode_ug/wer_8

 # Here is the baseline... we're slightly worse than the baseline on both test scenarios.
 #for x in exp/dnn4b_nnet2/decode*; do grep WER $x/wer_* | utils/best_wer.sh ; done
 #%WER 1.58 [ 198 / 12533, 22 ins, 45 del, 131 sub ] exp/dnn4b_nnet2/decode/wer_3
 #%WER 7.80 [ 977 / 12533, 83 ins, 151 del, 743 sub ] exp/dnn4b_nnet2/decode_ug/wer_6
)



