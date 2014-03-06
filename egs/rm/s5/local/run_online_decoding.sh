#!/bin/bash

. cmd.sh


steps/online/prepare_online_decoding.sh --cmd "$train_cmd" data/train data/lang \
    exp/tri3b exp/tri3b_mmi/final.mdl exp/tri3b_online/ || exit 1;

steps/online/decode.sh --config conf/decode.config --cmd "$decode_cmd" --nj 20 exp/tri3b/graph \
  data/test exp/tri3b_online/decode

steps/online/decode.sh --do-endpointing true \
  --config conf/decode.config --cmd "$decode_cmd" --nj 20 exp/tri3b/graph \
  data/test exp/tri3b_online/decode_endpointing

steps/online/decode.sh --per-utt true --config conf/decode.config \
   --cmd "$decode_cmd" --nj 20 exp/tri3b/graph \
  data/test exp/tri3b_online/decode_per_utt

# grep WER exp/tri3b_online/decode/wer_* | utils/best_wer.sh 
# %WER 2.06 [ 258 / 12533, 29 ins, 46 del, 183 sub ] exp/tri3b_online/decode/wer_10

# grep WER exp/tri3b_online/decode_endpointing/wer_* | utils/best_wer.sh 
# %WER 2.07 [ 260 / 12533, 33 ins, 46 del, 181 sub ] exp/tri3b_online/decode_endpointing/wer_10

# Treating each one as a separate utterance, we get this:
# grep WER exp/tri3b_online/decode_per_utt/wer_* | utils/best_wer.sh
# %WER 2.37 [ 297 / 12533, 41 ins, 56 del, 200 sub ] exp/tri3b_online/decode_per_utt/wer_9

# The baseline WER is:
# %WER 1.92 [ 241 / 12533, 28 ins, 39 del, 174 sub ] exp/tri3b_mmi/decode/wer_4


# You can ignore the folowing; these were when I was debugging a difference between the
# online and non-online decoding, the commands may be useful as examples.
# cat exp/tri3b_online/decode/log/decode.*.log  | grep _ | grep -v LOG | grep -v gz | sort > foo
# cat exp/tri3b_online/decode_endpointing/log/decode.*.log  | grep _ | grep -v LOG | grep -v gz | sort > bar
# diff foo bar
#gunzip -c exp/tri3b_online/decode/lat.*.gz | lattice-1best ark:- ark:- | lattice-copy ark:- ark:- | nbest-to-linear ark:- ark,t:- | grep rkm05_st0619_oct87 | show-alignments data/lang/phones.txt exp/tri3b/final.mdl ark:-
#gunzip -c exp/tri3b_online/decode_endpointing/lat.*.gz | lattice-1best ark:- ark:- | lattice-copy ark:- ark:- | nbest-to-linear ark:- ark,t:- | grep rkm05_st0619_oct87 | show-alignments data/lang/phones.txt exp/tri3b/final.mdl ark:-
# gunzip -c exp/tri3b_online/decode_endpointing/lat.*.gz | lattice-copy ark:- ark:- | lattice-to-fst ark:-  "scp,p,t:echo rkm05_st0619_oct87 -|" | utils/int2sym.pl -f 3- data/lang/words.txt
