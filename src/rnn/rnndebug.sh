#!/bin/bash
olddir=$PWD
cd /mnt/matylda5/kombrink/src/kaldi/trunk/egs/wsj/s1/
. path.sh
cd $olddir

N=10
S=0.0625

for L in 1.0 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1 0.0; do

#echo "WER KN5 (BASELINE)"
#compute-wer --text --mode=present ark:/mnt/matylda5/kombrink/src/kaldi/trunk/egs/wsj/s1/exp/decode_tri1ckn_latgen_tgpr_eval92/test_trans.filt ark:1best.kn5.txt

echo "RESCORE WITH RNN"
time ./rnn-rescore-kaldi --lambda=$L --acoustic-scale=$S --n=$N words.txt ark,t:"gunzip -c /mnt/matylda5/kombrink/src/kaldi/trunk/egs/wsj/s1/lmrescore/lats.newlm.gz|" WSJ.35M.200cl.350h.kaldi.rnn ark,t:rnn.92.lats

echo "EXTRACT NEW ONEBEST"
lattice-best-path --acoustic-scale=$S ark:rnn.92.lats ark,t:- 2> /dev/null | /mnt/matylda5/kombrink/src/kaldi/trunk/egs/wsj/s1/scripts/int2sym.pl --ignore-first-field words.txt | sed 's/ <s>//' | sed 's/ <\/s>//' > rnntest 

echo "WER RNN*$L"
compute-wer --text --mode=present ark:/mnt/matylda5/kombrink/src/kaldi/trunk/egs/wsj/s1/exp/decode_tri1ckn_latgen_tgpr_eval92/test_trans.filt ark:rnntest

done
