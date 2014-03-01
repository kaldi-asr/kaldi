#!/bin/bash

. cmd.sh


steps/online/prepare_online_decoding.sh --cmd "$train_cmd" data/train data/lang \
    exp/tri3b exp/tri3b_mmi/final.mdl exp/tri3b_online/

steps/online/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 20 exp/tri3b/graph \
  data/test exp/tri3b_online/decode

steps/online/decode.sh --do-endpointing true \
  --config conf/decode.config --cmd "$decode_cmd" --nj 20 exp/tri3b/graph \
  data/test exp/tri3b_online/decode_endpointing

steps/online/decode.sh --per-utt true --config conf/decode.config \
   --cmd "$decode_cmd" --nj 20 exp/tri3b/graph \
  data/test exp/tri3b_online/decode_per_utt

# grep WER exp/tri3b_online/decode/wer_* | utils/best_wer.sh 
# %WER 2.00 [ 251 / 12533, 28 ins, 45 del, 178 sub ] exp/tri3b_online/decode/wer_10

# grep WER exp/tri3b_online/decode_endpointing/wer_* | utils/best_wer.sh 
# %WER 2.27 [ 284 / 12533, 61 ins, 49 del, 174 sub ] exp/tri3b_online/decode_endpointing/wer_12

# Treating each one as a separate utterance, we get this:
# grep WER exp/tri3b_online/decode_per_utt/wer_* | utils/best_wer.sh
# %WER 2.37 [ 297 / 12533, 41 ins, 56 del, 200 sub ] exp/tri3b_online/decode_per_utt/wer_9

# The baseline WER is:
# %WER 1.92 [ 241 / 12533, 28 ins, 39 del, 174 sub ] exp/tri3b_mmi/decode/wer_4

