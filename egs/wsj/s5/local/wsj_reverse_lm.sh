#!/bin/bash

# Copyright 2012  Brno University of Technology (Author: Mirko Hannemann)
# Apache 2.0

. ./path.sh || exit 1;

srcdir=data/local/data
lmdir=data/local/nist_lm
tmpdir=data/local/lm_tmp
lexicon=data/local/lang_tmp.reverse/lexicon.txt
mkdir -p $tmpdir

# For each language model, create the corresponding FST
# and the corresponding lang_test_* directory.

echo Preparing language models for test

for lm_suffix in bg_5k; do
  test=data/lang_test_${lm_suffix}.reverse
  test_fwd=data/lang_test_${lm_suffix}
  mkdir -p $test
  for f in phones.txt words.txt phones.txt L.fst L_disambig.fst \
     phones/; do
    cp -r data/lang.reverse/$f $test
  done
  gunzip -c $lmdir/lm_${lm_suffix}.arpa.gz | \
   utils/find_arpa_oovs.pl $test/words.txt  > $tmpdir/oovs_${lm_suffix}.txt

  # grep -v '<s> <s>' because the LM seems to have some strange and useless
  # stuff in it with multiple <s>'s in the history.  Encountered some other similar
  # things in a LM from Geoff.  Removing all "illegal" combinations of <s> and </s>,
  # which are supposed to occur only at being/end of utt.  These can cause 
  # determinization failures of CLG [ends up being epsilon cycles].
  gunzip -c $lmdir/lm_${lm_suffix}.arpa.gz | \
    grep -v '<s> <s>' | \
    grep -v '</s> <s>' | \
    grep -v '</s> </s>' | \
    arpa2fst --reverse - | fstprint | \
    utils/remove_oovs.pl $tmpdir/oovs_${lm_suffix}.txt | \
    utils/eps2disambig.pl | utils/s2eps.pl | fstcompile --isymbols=$test/words.txt \
      --osymbols=$test/words.txt  --keep_isymbols=false --keep_osymbols=false \
      --arc_type=log | fstrmepsilon > $test/G_log.fst

  #fstreverse $test/G_log.fst > $test/G_log_rev.fst
  echo "Push weights in log semi-ring: can take lots of time!"
  # delta must be very small otherwise weight pushing won't succeed
  fstpush --push_weights=true --push_labels=true --delta=1E-7 $test/G_log.fst >$test/G_log_pushed.fst
  #fstproject --project_output=true > $test/G_log_pushed.fst
  echo "weight pushing done"
  #convert to tropical semiring
  fstprint --isymbols=$test/words.txt --osymbols=$test/words.txt $test/G_log_pushed.fst | \
#    sed 's#<s>#<ggg>#g' | sed 's#</s>#<s>#g' | sed 's#<ggg>#</s>#g'  > $test/G_log_pushed.txt #| \
    fstcompile --isymbols=$test/words.txt --osymbols=$test/words.txt \
     --keep_isymbols=false --keep_osymbols=false > $test/G.fst
  fstisstochastic $test/G.fst
 # The output is like:
 # 9.14233e-05 -0.259833
 # we do expect the first of these 2 numbers to be close to zero (the second is
 # nonzero because the backoff weights make the states sum to >1).
 # Because of the <s> fiasco for these particular LMs, the first number is not
 # as close to zero as it could be.

  testset=local/wsj_test.txt
  echo "perplexity using",$test_fwd/G.fst
  #cat $testset | awk '{fn="sent"NR; print"0 1 <s>" >> fn; s=1; for(i=1;i<=NF;i++){print s,s+1,$i >> fn; s=s+1;} print s,s+1,"</s>" >> fn; print s+1 >> fn; close(fn);}'
  cat $testset | awk '{fn="sent"NR; s=0; for(i=1;i<=NF;i++){print s,s+1,$i >> fn; s=s+1;} print s >> fn; close(fn);}'
  sentlist='sent?' # sent??'
  for x in $sentlist; do
    y=$x.fst
    fstcompile --acceptor --isymbols=$test/words.txt $x > $y
    rm $x
    fstcompose $y $test_fwd/G.fst | fstprint --acceptor --isymbols=$test/words.txt | awk 'BEGIN {sum=0;words=0} {if (NF>2) {sum=sum+$4; if ($3!~"<eps>") { words++; }} else {sum=sum+$2}} END{print sum,words,exp(sum/words)}' >>ppx.txt
    rm $y
  done
  paste ppx.txt $testset
  rm ppx.txt

  echo "perplexity using",$test/G.fst
  #cat $testset | awk '{fn="tnes"NR; print"0 1 <s>" >> fn; s=1; for(i=NF;i>0;i--){print s,s+1,$i >> fn; s=s+1;} print s,s+1,"</s>" >> fn; print s+1 >> fn; close(fn);}'
  cat $testset | awk '{fn="tnes"NR; s=0; for(i=NF;i>0;i--){print s,s+1,$i >> fn; s=s+1;} print s >> fn; close(fn);}'
  sentlist='tnes?' # tnes??'
  for x in $sentlist; do
    y=$x.fst
    fstcompile --acceptor --isymbols=$test/words.txt $x > $y
    rm $x
    fstcompose $y $test/G.fst | fstprint --acceptor --isymbols=$test/words.txt | awk 'BEGIN {sum=0;words=0} {if (NF>2) {sum=sum+$4; if ($3!~"<eps>") { words++; }} else {sum=sum+$2}} END{print sum,words,exp(sum/words)}' >>ppx.txt
    rm $y
  done
  paste ppx.txt $testset
  rm ppx.txt

  # Everything below is only for diagnostic.
  # Checking that G has no cycles with empty words on them (e.g. <s>, </s>);
  # this might cause determinization failure of CLG.
  # #0 is treated as an empty word.
  mkdir -p $tmpdir/g
  awk '{if(NF==1){ printf("0 0 %s %s\n", $1,$1); }} END{print "0 0 #0 #0"; print "0";}' \
    < "$lexicon"  >$tmpdir/g/select_empty.fst.txt
  fstcompile --isymbols=$test/words.txt --osymbols=$test/words.txt $tmpdir/g/select_empty.fst.txt | \
   fstarcsort --sort_type=olabel | fstcompose - $test/G.fst > $tmpdir/g/empty_words.fst
  fstinfo $tmpdir/g/empty_words.fst | grep cyclic | grep -w 'y' && 
    echo "Language model has cycles with empty words" && exit 1
  rm -r $tmpdir/g
done

echo "Succeeded in formatting data."
rm -r $tmpdir
