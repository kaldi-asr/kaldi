#!/bin/bash

. cmd.sh


steps/online/prepare_online_decoding.sh --add-pitch true --cmd "$train_cmd" data/train data/lang \
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
# %WER 3.20 [ 401 / 12533, 67 in , 50 del, 284  ub ] exp/tri3b_online/decode/wer_13

# grep WER exp/tri3b_online/decode_endpointing/wer_* | utils/best_wer.sh
# %WER 3.21 [ 402 / 12533, 73 in , 48 del, 281  ub ] exp/tri3b_online/decode_endpointing/wer_13

# Treating each one as a separate utterance, we get this:
# grep WER exp/tri3b_online/decode_per_utt/wer_* | utils/best_wer.sh
# %WER 3.62 [ 454 / 12533, 80 in , 58 del, 316  ub ] exp/tri3b_online/decode_per_utt/wer_13

# The baseline WER is:
# %WER 2.11 [ 265 / 12533, 44 in , 40 del, 181  ub ] exp/tri3b_mmi/decode/wer_7

# You can ignore the folowing; these were when I was debugging a difference between the
# online and non-online decoding, the commands may be useful as examples.
# cat exp/tri3b_online/decode/log/decode.*.log  | grep _ | grep -v LOG | grep -v gz | sort > foo
# cat exp/tri3b_online/decode_endpointing/log/decode.*.log  | grep _ | grep -v LOG | grep -v gz | sort > bar
# diff foo bar
#gunzip -c exp/tri3b_online/decode/lat.*.gz | lattice-1best ark:- ark:- | lattice-copy ark:- ark:- | nbest-to-linear ark:- ark,t:- | grep rkm05_st0619_oct87 | show-alignments data/lang/phones.txt exp/tri3b/final.mdl ark:-
#gunzip -c exp/tri3b_online/decode_endpointing/lat.*.gz | lattice-1best ark:- ark:- | lattice-copy ark:- ark:- | nbest-to-linear ark:- ark,t:- | grep rkm05_st0619_oct87 | show-alignments data/lang/phones.txt exp/tri3b/final.mdl ark:-
# gunzip -c exp/tri3b_online/decode_endpointing/lat.*.gz | lattice-copy ark:- ark:- | lattice-to-fst ark:-  "scp,p,t:echo rkm05_st0619_oct87 -|" | utils/int2sym.pl -f 3- data/lang/words.txt
