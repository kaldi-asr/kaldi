#!/bin/bash

. path.sh || exit 1;
export PATH=../../../src/latbin:../../../src/rnn:../../../src/bin:$PATH

#TMP=$(mktemp)
TMP=t
mkdir -p $TMP
echo "Using $TMP"
#pushd $TMP
cd t

cp /homes/kazi/kombrink/WSJ.tar.gz .
tar xzf WSJ.tar.gz
gunzip lats.newlm.gz

cd ..
#popd

NBEST=10
INLATS=$TMP/lats.newlm
NG_LM="/mnt/matylda5/kombrink/EXP/WSJ-rnn/baseline_LM/srilm.o3g.kn.gz"
RNN_LM=$TMP/WSJ.35M.200cl.350h.kaldi.rnn
WORDSYM=$TMP/words.txt
TRANS=$TMP/eval_nov92.txt

AMS=0.0625

. path.sh
export PATH=../../../src/latbin:$PATH

echo "Compiling LM fst"
  gunzip -c $NG_LM | \
    grep -v '<s> <s>' | \
    grep -v '</s> <s>' | \
    grep -v '</s> </s>' | \
    arpa2fst - | fstprint | \
    scripts/remove_oovs.pl data/oovs_srilm.3g.kn.gz.txt | \
    scripts/eps2disambig.pl |  fstcompile --isymbols=$WORDSYM --osymbols=$WORDSYM \
     --keep_isymbols=false --keep_osymbols=false > $TMP/G.fst

echo "Extracting $NBEST best"

lattice-nbest --acoustic-scale=$AMS --n=$NBEST ark:$INLATS ark:- | lattice-lmrescore --lm-scale=-1 ark:- $TMP/G.fst ark,t:$TMP/lats.nbest

echo "Rescoring"

rnn-rescore --rnn_scale=1 $WORDSYM $RNN_LM ark:$TMP/lats.nbest $RNN_LM ark:$TMP/lats.nbest.rnn

echo "WER"
lattice-best-path --acoustic-scale=$AMS ark:$TMP/lats.nbest.rnn ark,t:- 2> /dev/null | scripts/int2sym.pl --ignore-first-field $WORDSYM | sed 's/ <s>//' | sed 's/ <\/s>//' > rnntest

compute-wer --text --mode=present "ark:cat $TRANS | sed 's:<NOISE>::g' | sed 's:<SPOKEN_NOISE>::g' |" ark:rnntest

