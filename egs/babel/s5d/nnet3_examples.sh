# The results shown below are for Telugu fullLP condition
#TDNN
  local/nnet3/run_tdnn.sh \
    --affix "6layer_r512" \
    --splice-indexes "-2,-1,0,1,2 -1,2 -3,3 -7,2 0 0 "  \
    --relu-dim 512 || exit 1;
 
  # I modified the TDNN scripts to run for 5 epochs, however these results are with 3 epoch training 
  ./run-4-anydecode.sh --skip-kws true --dir dev10h.seg --nnet3-model nnet3/tdnn_6layer_r512_sp
  #%WER 68.4 | 22131 40145 | 36.3 45.9 17.9 4.7 68.4 31.9 | -1.082 | exp/nnet3/tdnn_6layer_r512_sp/decode_dev10h.seg/score_10/dev10h.seg.ctm.sys
  ./run-4-anydecode.sh --skip-kws true --dir dev10h.pem --nnet3-model nnet3/tdnn_6layer_r512_sp
  #%WER 67.1 | 22131 40145 | 36.4 45.9 17.8 3.5 67.1 29.6 | -0.902 | exp/nnet3/tdnn_6layer_r512_sp/decode_dev10h.pem/score_11/dev10h.pem.ctm.sys




#LSTM 
  local/nnet3/run_lstm.sh

  ./run-4-anydecode.sh --skip-kws true --dir dev10h.seg --is-rnn true --nnet3-model nnet3/lstm_sp --extra-left-context 40 --frames-per-chunk 20
  #%WER 68.0 | 22131 40145 | 38.2 44.8 17.0 6.2 68.0 33.5 | -1.491 | exp/nnet3/lstm_sp/decode_dev10h.seg/score_10/dev10h.seg.ctm.sys
  ./run-4-anydecode.sh --skip-kws true --dir dev10h.pem --is-rnn true --nnet3-model nnet3/lstm_sp --extra-left-context 40 --frames-per-chunk 20
  #%WER 65.1 | 22131 40145 | 39.2 45.9 14.9 4.3 65.1 28.8 | -1.299 | exp/nnet3/lstm_sp/decode_dev10h.pem/score_10/dev10h.pem.ctm.sys 


#BLSTM 
  local/nnet3/run_blstm.sh 
  ./run-4-anydecode.sh --skip-kws true --dir dev10h.seg --is-rnn true --nnet3-model nnet3/lstm_bidirectional_sp --extra-left-context 40 --extra-right-context 40 --frames-per-chunk 20
  #%WER 67.1 | 22131 40145 | 38.8 44.9 16.3 5.9 67.1 33.6 | -1.737 | exp/nnet3/lstm_birectional_cell512_sp/decode_dev10h.seg/score_10/dev10h.seg.ctm.sys
  ./run-4-anydecode.sh --skip-kws true --dir dev10h.pem --is-rnn true --nnet3-model nnet3/lstm_bidirectional_sp --extra-left-context 40 --extra-right-context 40 --frames-per-chunk 20
  #%WER 64.2 | 22131 40145 | 39.8 46.0 14.2 4.0 64.2 29.0 | -1.548 | exp/nnet3/lstm_birectional_cell512_sp/decode_dev10h.pem/score_10/dev10h.pem.ctm.sys

