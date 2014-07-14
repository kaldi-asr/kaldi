#!/bin/bash

. cmd.sh


# For now we're demonstrating this script with the models in exp/nnet4b or
# exp/nnet4b_gpu, which are trained with multiple-VTLN-warped data, but in
# principle this script would work with any type of neural net input

if [ -f exp/nnet4b/final.mdl ]; then
  srcdir=exp/nnet4b_gpu
elif [ -f exp/nnet4b_gpu/final.mdl ]; then
  srcdir=exp/nnet4b
else
  echo "$0: before running this script, please run local/nnet2/run_4b.sh or local/nnet2/run_4b_gpu.sh"
fi
  

mkdir -p exp/nnet2_online

steps/online/nnet2/train_diag_ubm.sh --cmd "$train_cmd" --nj 10 --num-frames 200000 \
   data/train 512 exp/tri3b exp/nnet2_online/diag_ubm

steps/online/nnet2/train_ivector_extractor.sh --cmd "$train_cmd" --nj 4 \
   data/train exp/nnet2_online/diag_ubm exp/nnet2_online/extractor

steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 4 \
   data/train exp/nnet2_online/extractor exp/nnet2_online/ivectors

dir=exp/nnet2_online/nnet


steps/nnet2/train_pnorm_fast.sh --stage $train_stage \
   --num-threads "$num_threads" \
   --minibatch-size "$minibatch_size" \
   --parallel-opts "$parallel_opts" \
   --num-jobs-nnet 4 \
   --num-epochs-extra 10 --add-layers-period 1 \
   --num-hidden-layers 2 \
   --mix-up 4000 \
   --initial-learning-rate 0.02 --final-learning-rate 0.004 \
   --cmd "$decode_cmd" \
   --pnorm-input-dim 1000 \
   --pnorm-output-dim 200 \
    data/train data/lang exp/tri3b_ali $dir  




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
