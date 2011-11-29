#!/bin/bash

if [ -f ./path.sh ]; then . ./path.sh; fi

if [ $# -ne 3 ]; then
   echo "Usage: scripts/score_lats_ctm.sh <decode-dir> <lang-dir> <data-dir>"
   echo "e.g.: scripts/score_lats_ctm.sh exp/tri5a/decode_eval2000 data/lang_test/ data/eval2000/"
   exit 1;
fi

dir=$1
lang=$2
data=$3

model=$dir/../final.mdl # assume model one level up from decoding dir.

hubscr=../../../tools/sctk-2.4.0/bin/hubscr.pl 
export PATH=$PATH:`dirname $hubscr`

for f in "$hubscr" $data/stm $data/glm $lang/words.txt $lang/phones_disambig.txt \
      $lang/L_align.fst $model $data/segments; do
  [ ! -f $f ] && echo "score_lats_ctm.sh: expecting file $f to exist" && exit 1;
done

wbegin=`grep "#1" $lang/phones_disambig.txt | head -1 | awk '{print $2}'`
wend=`grep "#2" $lang/phones_disambig.txt | head -1 | awk '{print $2}'`

[ ! -n "$wbegin" ] && echo "Error with word-begin symbol (bad phones_disambig.txt?)" && exit 1
[ ! -n "$wend" ] && echo "Error with word-end symbol (bad phones_disambig.txt?)" && exit 1

rm $dir/.error 2>/dev/null
for group in "9 10 11" "12 13 14" "15 16"; do # do the rescoring in batches of up to 3.
  for inv_acwt in $group; do
   (    
    mkdir -p $dir/score_${inv_acwt}
    acwt=`perl -e "print (1.0/$inv_acwt);"`
    # Since we'll need the word aligment, get the state-level alignment
    # as well as the word-level one, for each acwt.
    lattice-best-path --acoustic-scale=$acwt --word-symbol-table=$symtab \
     "ark:gunzip -c $dir/lat.*.gz|" "ark,t:|gzip -c >$dir/score_${inv_acwt}/tra.gz" \
      "ark,t:|gzip -c >$dir/score_${inv_acwt}/ali.gz" 2>$dir/score_${inv_acwt}/rescore.log || exit 1;
   ) &
  done
  wait
  [ -f $dir/.error ] && \
     echo "score_lats_ctm.sh: error rescoring lattices; look into logs in $dir/score_*/rescore.log for more details"  && \
     exit 1;
  for inv_acwt in $group; do
    name=`basename $data` # e.g. "eval2000"
    # Create ctm this pipe first creates a ctm that's relative to the utterance-ids,
    # and then makes it relative to the conversation sides).
   ! ( ali-to-phones $model "ark:gunzip -c $dir/score_${inv_acwt}/ali.gz|" ark:- | \
     phones-to-prons $lang/L_align.fst $wbegin $wend ark:- "ark:gunzip -c $dir/score_${inv_acwt}/tra.gz|" ark,t:- | \
     prons-to-wordali ark:- \
    "ark:ali-to-phones --write-lengths $model 'ark:gunzip -c $dir/score_${inv_acwt}/ali.gz|' ark,t:- |" ark,t:- | \
     scripts/wali_to_ctm.sh - $lang/words.txt $data/segments | grep -v -E '\[NOISE|LAUGHTER|VOCALIZED-NOISE\]' | \
     grep -v -E '<UNK>|%HESITATION' )  > $dir/score_${inv_acwt}/$name.ctm  2>$dir/score_${inv_acwt}/log && \
      echo "score_lats_ctm.sh: error generating ctm, see $dir/score_${inv_acwt}/log" && exit 1;
   
    ! $hubscr -V -l english -h hub5 -g $data/glm -r $data/stm $dir/score_${inv_acwt}/${name}.ctm \
       >&$dir/score_${inv_acwt}/sclite.log && \
      echo "score_lats_ctm.sh: error doing sclite scoring, see $dir/score_${inv_acwt}/sclite.log" \
      && exit 1
  done
done

exit 0
