#!/bin/bash

#
if [ -f WSJ.tar.gz ]; then
  echo "data archive found!"
else
  echo "trying to download the data archive using wget"
  wget http://www.fit.vutbr.cz/~kombrink/personal/rnn-kaldi/WSJ.tar.gz
fi;

tar xzf WSJ.tar.gz

N=10
S=0.0625
INLATS=lats.newlm.gz

# RNN lm rescoring
for L in 0.0 0.25 0.5 0.75 1.0; do

echo "RESCORE WITH RNN"
time ./rnn-rescore-kaldi --lambda=$L --acoustic-scale=$S --n=$N words.txt ark,t:"gunzip -c $INLATS |" WSJ.35M.200cl.350h.kaldi.rnn ark,t:rnn.92.lats

echo "EXTRACT NEW ONEBEST"
../latbin/lattice-best-path --acoustic-scale=$S ark:rnn.92.lats ark,t:- 2> /dev/null | ../../egs/wsj/s1/scripts/int2sym.pl --ignore-first-field words.txt | sed 's/ <s>//' | sed 's/ <\/s>//' > rnntest 

echo "WER RNN*$L"
../bin/compute-wer --text --mode=present "ark:cat eval_nov92.txt | sed 's:<NOISE>::g' | sed 's:<SPOKEN_NOISE>::g' |" ark:rnntest

done


echo "LATTICE ORACLE ERROR"


# these symbols we want to ignore completely
echo "<UNK> <s> </s>" | tr ' ' '\n' > ignore.txt
# construct an FST containing the reference
cat eval_nov92.txt | sed 's:<NOISE>::g' | sed 's:<SPOKEN_NOISE>::g'| ../../egs/wsj/s1/scripts/sym2int.pl --ignore-first-field words.txt | ../latbin/string-to-lattice ark,t:eval92.lats
# we need to prune here a bit due to memory consumption! Hence, oracle error of the real lattices is even lower!
# the lower bound is defined by the OOV rate (=1.9%)
../latbin/lattice-prune --acoustic-scale=$S --beam=7 ark:"gunzip -c $INLATS |" ark,t:"| gzip -c > ${INLATS}.pruned" 2> /dev/null
../latbin/lattice-oracle --word-symbol-table=words.txt --wildcard-symbols-list=ignore.txt ark:eval92.lats ark:"gunzip -c ${INLATS}.pruned |" ark,t:oracle.pruned.tra 2> /dev/null
cat eval_nov92.txt | sed 's:<NOISE>::g' | sed 's:<SPOKEN_NOISE>::g' | ../../egs/wsj/s1/scripts/sym2int.pl --ignore-first-field words.txt | ../bin/compute-wer ark,t:- ark:oracle.pruned.tra

echo "${N}-BEST ORACLE ERROR"
# now compare how well the n-bests can do:
../latbin/lattice-nbest --n=$N --acoustic-scale=$S ark:"gunzip -c $INLATS |" ark,t:"| gzip -c > ${INLATS}.n$N" 2> /dev/null
../latbin/lattice-oracle --word-symbol-table=words.txt --wildcard-symbols-list=ignore.txt ark:eval92.lats ark:"gunzip -c ${INLATS}.n$N |" ark,t:oracle.n$N.tra 2> /dev/null
cat eval_nov92.txt | sed 's:<NOISE>::g' | sed 's:<SPOKEN_NOISE>::g' | ../../egs/wsj/s1/scripts/sym2int.pl --ignore-first-field words.txt | ../bin/compute-wer ark,t:- ark:oracle.n$N.tra

