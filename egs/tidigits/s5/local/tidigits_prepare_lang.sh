#!/usr/bin/env bash

# Copyright 2012 Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0.

# This script prepares the lang/ directory.
#

. ./path.sh 


# Decided to do this using something like a real lexicon, although we
# could also have used whole-word models.
tmpdir=data/local/dict
lang=data/lang
mkdir -p $tmpdir

cat >$tmpdir/lexicon.txt <<EOF
z z iy r ow
o ow
1 w ah n
2 t uw
3 th r iy
4 f ao r
5 f ay v
6 s ih k s
7 s eh v ah n
8 ey t
9 n ay n
EOF
# and note, we'll have a silence phone, but it won't appear
# in this form of lexicon as there's no silence word; it's an option
# in the lexicon FST that gets added by the script.

mkdir -p $lang/phones

# symbol-table for words:
cat $tmpdir/lexicon.txt | awk '{print $1}' | awk 'BEGIN {print "<eps> 0"; n=1;} { printf("%s %s\n", $1, n++); }' \
  >$lang/words.txt

# list of phones.
cat $tmpdir/lexicon.txt | awk '{for(n=2;n<=NF;n++) seen[$n]=1; } END{print "sil"; for (w in seen) { print w; }}' \
 >$tmpdir/phone.list

# symbol-table for phones:
cat $tmpdir/phone.list | awk 'BEGIN {print "<eps> 0"; n=1;} { printf("%s %s\n", $1, n++); }' \
  >$lang/phones.txt

p=$lang/phones
echo sil > $p/silence.txt
echo sil > $p/context_indep.txt
echo sil > $p/optional_silence.txt
grep -v -w sil $tmpdir/phone.list > $p/nonsilence.txt
touch $p/disambig.txt # disambiguation-symbols list, will be empty.
touch $p/extra_questions.txt # list of "extra questions"-- empty; we don't
 # have things like tone or word-positions or stress markings.
cat $tmpdir/phone.list > $p/sets.txt # list of "phone sets"-- each phone is in its
 # own set.  Normally, each line would have a bunch of word-position-dependenent or
 # stress-dependent realizations of the same phone.

for t in silence nonsilence context_indep optional_silence disambig; do
  utils/sym2int.pl $lang/phones.txt <$p/$t.txt >$p/$t.int
  cat $p/$t.int | awk '{printf(":%d", $1);} END{printf "\n"}' | sed s/:// > $p/$t.csl 
done
for t in extra_questions sets; do
  utils/sym2int.pl $lang/phones.txt <$p/$t.txt >$p/$t.int
done

cat $tmpdir/phone.list | awk '{printf("shared split %s\n", $1);}' >$p/roots.txt
utils/sym2int.pl -f 3-  $lang/phones.txt $p/roots.txt >$p/roots.int

echo z > $lang/oov.txt # we map OOV's to this.. there are no OOVs in this setup,
   # but the scripts expect this file to exist.
utils/sym2int.pl $lang/words.txt <$lang/oov.txt >$lang/oov.int

# Note: "word_boundary.{txt,int}" will not exist in this setup.  This will mean it's
# not very easy to get word alignments, but it simplifies some things.

# Make the FST form of the lexicon (this includes optional silence).
utils/make_lexicon_fst.pl $tmpdir/lexicon.txt 0.5 sil | \
  fstcompile --isymbols=$lang/phones.txt --osymbols=$lang/words.txt \
  --keep_isymbols=false --keep_osymbols=false | \
   fstarcsort --sort_type=olabel > $lang/L.fst 

# Note: in this setup there are no "disambiguation symbols" because the lexicon
# contains no homophones; and there is no '#0' symbol in the LM because it's
# not a backoff LM, so L_disambig.fst is the same as L.fst.

cp $lang/L.fst $lang/L_disambig.fst

num_sil_states=5
num_nonsil_states=3
silphonelist=`cat $lang/phones/silence.csl`
nonsilphonelist=`cat $lang/phones/nonsilence.csl`
utils/gen_topo.pl $num_nonsil_states $num_sil_states $nonsilphonelist $silphonelist >$lang/topo

# Now we prepare a simple grammar G.fst that's a kind of loop of
# digits (no silence in this, since that's handled in L.fst)
# there are 12 options: 1-9, zero, oh, and end-of-sentence.
penalty=`perl -e '$prob = 1.0/12; print -log($prob); '` # negated log-prob,
  # which becomes the cost on the FST.
( for x in `echo z o 1 2 3 4 5 6 7 8 9`; do
   echo 0 0 $x $x $penalty   # format is: from-state to-state input-symbol output-symbol cost
 done 
 echo 0 $penalty # format is: state final-cost
) | fstcompile --isymbols=$lang/words.txt --osymbols=$lang/words.txt \
   --keep_isymbols=false --keep_osymbols=false |\
   fstarcsort --sort_type=ilabel > $lang/G.fst


exit 0;








if [ $# -ne 0 ]; then
   echo "Argument should be the TIDIGITS directory, see ../run.sh for example."
   exit 1;
fi




tidigits=$1

tmpdir=`pwd`/data/local/data
mkdir -p $tmpdir

# Note: the .wav files are not in .wav format but "sphere" format (this was 
# produced in the days before Windows).

find $tidigits/tidigits/train -name '*.wav' > $tmpdir/train.flist
n=`cat $tmpdir/train.flist | wc -l`
[ $n -eq 8623 ] || echo Unexpected number of training files $n versus 8623

find $tidigits/tidigits/test -name '*.wav' > $tmpdir/test.flist
n=`cat $tmpdir/test.flist | wc -l`
[ $n -eq 8700 ] || echo Unexpected number of test files $n versus 8700

sph2pipe=$KALDI_ROOT/tools/sph2pipe_v2.5/sph2pipe
if [ ! -x $sph2pipe ]; then
   echo "Could not find (or execute) the sph2pipe program at $sph2pipe";
   exit 1;
fi

for x in train test; do
  # get scp file that has utterance-ids and maps to the sphere file.
  cat $tmpdir/$x.flist | perl -ane 'm|/(..)/([1-9zo]+[ab])\.wav| || die "bad line $_"; print "$1_$2 $_"; ' \
   | sort > $tmpdir/${x}_sph.scp
  # turn it into one that has a valid .wav format in the modern sense (i.e. RIFF format, not sphere).
  # This file goes into its final location
  mkdir -p data/$x
  awk '{printf("%s '$sph2pipe' -f wav %s |\n", $1, $2);}' < $tmpdir/${x}_sph.scp > data/$x/wav.scp

  # Now get the "text" file that says what the transcription is.
  cat data/$x/wav.scp | 
   perl -ane 'm/^(.._([1-9zo]+)[ab]) / || die; $text = join(" ", split("", $2)); print "$1 $text\n";' \
    <data/$x/wav.scp >data/$x/text

  # now get the "utt2spk" file that says, for each utterance, the speaker name.  
  perl -ane 'm/^((..)_\S+) / || die; print "$1 $2\n"; ' \
    <data/$x/wav.scp >data/$x/utt2spk
  # create the file that maps from speaker to utterance-list.
  utils/utt2spk_to_spk2utt.pl <data/$x/utt2spk >data/$x/spk2utt
done

echo "Data preparation succeeded"
