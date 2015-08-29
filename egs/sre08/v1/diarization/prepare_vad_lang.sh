#!/bin/bash
# Copyright 2012-2013  Johns Hopkins University (Author: Daniel Povey);
#                      Arnab Ghoshal
#                2014  Guoguo Chen
#                2015  Hainan Xu

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.

# This script prepares a directory such as data/lang/, in the standard format,
# given a source directory containing a dictionary lexicon.txt in a form like:
# word phone1 phone2 ... phoneN
# per line (alternate prons would be separate lines), or a dictionary with probabilities
# called lexiconp.txt in a form:
# word pron-prob phone1 phone2 ... phoneN
# (with 0.0 < pron-prob <= 1.0); note: if lexiconp.txt exists, we use it even if
# lexicon.txt exists.
# and also files silence_phones.txt, nonsilence_phones.txt, optional_silence.txt
# and extra_questions.txt
# Here, silence_phones.txt and nonsilence_phones.txt are lists of silence and
# non-silence phones respectively (where silence includes various kinds of 
# noise, laugh, cough, filled pauses etc., and nonsilence phones includes the 
# "real" phones.)
# In each line of those files is a list of phones, and the phones on each line 
# are assumed to correspond to the same "base phone", i.e. they will be 
# different stress or tone variations of the same basic phone.
# The file "optional_silence.txt" contains just a single phone (typically SIL) 
# which is used for optional silence in the lexicon.
# extra_questions.txt might be empty; typically will consist of lists of phones,
# all members of each list with the same stress or tone; and also possibly a 
# list for the silence phones.  This will augment the automtically generated 
# questions (note: the automatically generated ones will treat all the 
# stress/tone versions of a phone the same, so will not "get to ask" about 
# stress or tone).

# This script constructs a host of other
# derived files, that go in lang/.

# Begin configuration section.
num_sil_states=5
num_nonsil_states=3
# end configuration sections

. utils/parse_options.sh 

if [ $# -ne 3 ]; then 
  echo "usage: utils/prepare_lang.sh <dict-src-dir> <oov-dict-entry> <tmp-dir> <lang-dir>"
  echo "e.g.: utils/prepare_lang.sh data/local/dict <SPOKEN_NOISE> data/local/lang data/lang"
  echo "<dict-src-dir> should contain the following files:"
  echo " extra_questions.txt  lexicon.txt nonsilence_phones.txt  optional_silence.txt  silence_phones.txt"
  echo "See http://kaldi.sourceforge.net/data_prep.html#data_prep_lang_creating for more info."
  echo "options: "
  echo "     --num-sil-states <number of states>             # default: 5, #states in silence models."
  echo "     --num-nonsil-states <number of states>          # default: 3, #states in non-silence models."
  exit 1;
fi

srcdir=$1
tmpdir=$2
dir=$3
mkdir -p $dir $tmpdir $dir/phones

[ -f path.sh ] && . ./path.sh

! utils/validate_dict_dir.pl $srcdir && \
  echo "*Error validating directory $srcdir*" && exit 1;

if [[ ! -f $srcdir/lexicon.txt ]]; then
  echo "**Creating $dir/lexicon.txt from $dir/lexiconp.txt"
  perl -ape 's/(\S+\s+)\S+\s+(.+)/$1$2/;' < $srcdir/lexiconp.txt > $srcdir/lexicon.txt || exit 1;
fi
if [[ ! -f $srcdir/lexiconp.txt ]]; then
  echo "**Creating $srcdir/lexiconp.txt from $srcdir/lexicon.txt"
  perl -ape 's/(\S+\s+)(.+)/${1}1.0\t$2/;' < $srcdir/lexicon.txt > $srcdir/lexiconp.txt || exit 1;
fi

if ! utils/validate_dict_dir.pl $srcdir >&/dev/null; then
  utils/validate_dict_dir.pl $srcdir  # show the output.
  echo "Validation failed (second time)"
  exit 1;
fi
 
cp -f $srcdir/lexiconp.txt $tmpdir/lexiconp.txt 

cat $srcdir/silence_phones.txt $srcdir/nonsilence_phones.txt | \
  sed 's/ /\n/g' | awk '(NF>0){print}' > $tmpdir/phones
paste -d' ' $tmpdir/phones $tmpdir/phones > $tmpdir/phone_map.txt

mkdir -p $dir/phones  # various sets of phones...

cat $srcdir/{,non}silence_phones.txt | utils/apply_map.pl $tmpdir/phone_map.txt > $dir/phones/sets.txt
cat $dir/phones/sets.txt | awk '{print "shared", "split", $0;}' > $dir/phones/roots.txt

cat $srcdir/silence_phones.txt | utils/apply_map.pl $tmpdir/phone_map.txt | \
 awk '{for(n=1;n<=NF;n++) print $n;}' > $dir/phones/silence.txt
cat $srcdir/nonsilence_phones.txt | utils/apply_map.pl $tmpdir/phone_map.txt | \
 awk '{for(n=1;n<=NF;n++) print $n;}' > $dir/phones/nonsilence.txt

# if extra_questions.txt is empty, it's OK.
cat $srcdir/extra_questions.txt 2>/dev/null | utils/apply_map.pl $tmpdir/phone_map.txt \
  >$dir/phones/extra_questions.txt

# Create phone symbol table.
cat $dir/phones/{silence,nonsilence}.txt | \
  awk '{n=NR; print $1, n;}' > $dir/phones.txt 

cat $tmpdir/lexiconp.txt | awk '{print $1}' | sort | uniq  | awk '
  {
    if ($1 == "<s>") {
      print "<s> is in the vocabulary!" | "cat 1>&2"
      exit 1;
    }
    if ($1 == "</s>") {
      print "</s> is in the vocabulary!" | "cat 1>&2"
      exit 1;
    }
    printf("%s %d\n", $1, NR);
  }' > $dir/words.txt || exit 1;

# create $dir/phones/align_lexicon.{txt,int}.
# This is the new-new style of lexicon aligning.

# First remove pron-probs from the lexicon.
perl -ape 's/(\S+\s+)\S+\s+(.+)/$1$2/;' <$tmpdir/lexiconp.txt >$tmpdir/align_lexicon.txt

cat $tmpdir/align_lexicon.txt | \
 perl -ane '@A = split; print $A[0], " ", join(" ", @A), "\n";' | sort | uniq > $dir/phones/align_lexicon.txt

# create phones/align_lexicon.int
cat $dir/phones/align_lexicon.txt | utils/sym2int.pl -f 3- $dir/phones.txt | \
  utils/sym2int.pl -f 1-2 $dir/words.txt > $dir/phones/align_lexicon.int

# Create the basic L.fst without disambiguation symbols, for use
# in training. 

utils/make_lexicon_fst.pl --pron-probs $tmpdir/lexiconp.txt | \
  fstcompile --isymbols=$dir/phones.txt --osymbols=$dir/words.txt \
  --keep_isymbols=false --keep_osymbols=false | \
  fstarcsort --sort_type=olabel > $dir/L.fst || exit 1;

# Create these lists of phones in colon-separated integer list form too, 
# for purposes of being given to programs as command-line options.
for f in silence nonsilence; do
  utils/sym2int.pl $dir/phones.txt <$dir/phones/$f.txt >$dir/phones/$f.int
  utils/sym2int.pl $dir/phones.txt <$dir/phones/$f.txt | \
   awk '{printf(":%d", $1);} END{printf "\n"}' | sed s/:// > $dir/phones/$f.csl || exit 1;
done

for x in sets extra_questions; do
  utils/sym2int.pl $dir/phones.txt <$dir/phones/$x.txt > $dir/phones/$x.int || exit 1;
done

utils/sym2int.pl -f 3- $dir/phones.txt <$dir/phones/roots.txt \
   > $dir/phones/roots.int || exit 1;

silphonelist=`cat $dir/phones/silence.csl`
nonsilphonelist=`cat $dir/phones/nonsilence.csl`
diarization/gen_vad_topo.pl $num_nonsil_states $num_sil_states $nonsilphonelist $silphonelist >$dir/topo

rm -f $dir/L_disambig.fst 2>/dev/null
ln -s L.fst $dir/L_disambig.fst

num_words=`cat $dir/words.txt | wc -l 2> /dev/null` || exit 1
#prob=`perl -e "print log($num_words) / log(10)"`
prob=`perl -e "print log($[num_words+1])"`
while IFS=$'\n' read line; do
  word=`echo $line | awk '{print $1}'`
  echo 0 0 $word $word $prob
done < $dir/words.txt > $tmpdir/G.txt
echo 0 $prob >> $tmpdir/G.txt

fstcompile --isymbols=$dir/words.txt --osymbols=$dir/words.txt \
  --keep_isymbols=false --keep_osymbols=false \
  < $tmpdir/G.txt > $dir/G.fst || exit 1

#echo "$(basename $0): validating output directory"
#! utils/validate_lang.pl $dir && echo "$(basename $0): error validating output" &&  exit 1;


exit 0;

