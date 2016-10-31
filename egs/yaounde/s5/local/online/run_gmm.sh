#!/bin/bash
. cmd.sh
. ./path.sh
. ./conf/vars.sh
set -eu
set -o pipefail
. utils/parse_options.sh

steps/online/prepare_online_decoding.sh \
    --cmd "$train_cmd" \
    --add-pitch true \
    data/train_semi_supervised \
    data/lang \
    exp/tri5_semi_supervised \
    exp/tri5_semi_supervised/final.mdl \
    exp/tri5_semi_supervised_online || exit 1;

steps/online/decode.sh \
    --config conf/decode.config \
    --cmd "$decode_cmd" \
    --nj $nj \
    exp/tri5_semi_supervised/graph \
    data/test \
    exp/tri5_semi_supervised_online/decode_test || exit 1;

#  online decoding with endpointing
#   endpointing   at  end of  utterance, not  beginning
#like a dialog system over the phone
# waits till you finish saying something  then does something
# endpoint detection is configurable in various ways 
#  uses   decoder traceback  to endpoint
# whether  silence, etc
steps/online/decode.sh \
    --do-endpointing true \
    --config conf/decode.config \
    --cmd "$decode_cmd" \
    --nj $nj \
    exp/tri5_semi_supervised/graph \
    data/test \
    exp/tri5_semi_supervised_online/decode_endpointing || exit 1;

#  like basic online decoding above
# except  treat each utterance separately 
# do not carry forward speaker adaptation state from  previous utterance

steps/online/decode.sh \
    --per-utt true \
    --config conf/decode.config \
    --cmd "$decode_cmd" \
    --nj $nj \
    exp/tri5_semi_supervised/graph \
    data/test \
    exp/tri5_semi_supervised_online/decode_per_utt || exit 1;

exit

grep WER exp/tri5_semi_supervised_3_online/decode_test/wer_* | utils/best_wer.sh 
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
