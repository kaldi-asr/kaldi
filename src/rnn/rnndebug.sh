#!/bin/bash
N=100
S=0.0625
INLATS=lats.newlm.gz

for L in 0.5 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do

echo "RESCORE WITH RNN"
time ./rnn-rescore-kaldi --lambda=$L --acoustic-scale=$S --n=$N words.txt ark,t:"gunzip -c $INLATS |" WSJ.35M.200cl.350h.kaldi.rnn ark,t:rnn.92.lats

echo "EXTRACT NEW ONEBEST"
../latbin/lattice-best-path --acoustic-scale=$S ark:rnn.92.lats ark,t:- 2> /dev/null | ../../egs/wsj/s1/scripts/int2sym.pl --ignore-first-field words.txt | sed 's/ <s>//' | sed 's/ <\/s>//' > rnntest 

echo "WER RNN*$L"
../bin/compute-wer --text --mode=present "ark:cat ../../egs/wsj/s1/data/eval_nov92.txt | sed 's:<NOISE>::g' | sed 's:<SPOKEN_NOISE>::g' |" ark:rnntest

done
