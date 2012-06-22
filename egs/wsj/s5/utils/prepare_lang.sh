#!/bin/bash
# Copyright Johns Hopkins University (Author: Daniel Povey)  2012
# Apache 2.0

# This script prepares a directory such as data/lang/, in the standard format,
# given a source directory containing a dictionary lexicon.txt in a form like:
# word phone1 phone2 ... phonen
# per line (alternate prons would be separate lines).
# and also files silence_phones.txt and nonsilence_phones.txt, optional_silence.txt
# and extra_questions.txt
# Here, silence_phones.txt and nonsilence_phones.txt are lists of silence and
# non-silence phones respectively (where silence includes various kinds of noise,
# laugh, cough, filled pauses etc., and nonsilence phones includes the "real" phones.)
# on each line of those files is a list of phones, and the phones on each line are
# assumed to correspond to the same "base phone", i.e. they will be different stress
# or tone variations of the same basic phone.
# The file "optional_silence.txt" contains just a single phone (typically SIL) which
# is used for optional silence in the lexicon.
# extra_questions.txt might be empty; typically will consist of lists of phones, all
# members of each list with the same stress or tone or something; and also possibly 
# a list for
# the silence phones.  This will augment the automtically generated questions (note:
# the automatically generated ones will treat all the stress/tone versions of a phone
# the same, so will not "get to ask" about stress or tone).

# This script adds word-position-dependent phones and constructs a host of other
# derived files, that go in data/lang/.

# Begin configuration section.
sil_prob=0.5
num_sil_states=5
num_nonsil_states=3
share_silence_phones=false  # if true, then share pdfs of different silence phones
  # together.
# end configuration sections

. utils/parse_options.sh 

if [ $# -ne 4 ]; then 
  echo "usage: utils/prepare_lang.sh <dict-src-dir> <oov-dict-entry> <tmp-dir> <lang-dir>"
  echo "e.g.: utils/prepare_lang.sh data/local/dict <SPOKEN_NOISE> data/local/lang data/lang"
  echo "options: "
  echo "     --sil-prob <probability of silence>             # default: 0.5 [must have 0 < silprob < 1]"
  echo "     --num-sil-states <number of states>             # default: 5, #states in silence models."
  echo "     --num-nonsil-states <number of states>          # default: 3, #states in non-silence models."
  echo "     --share-silence-phones (true|false)             # default: false; if true, share pdfs of "
  echo "                                                     # all non-silence phones. "
  exit 1;
fi

srcdir=$1
oov_word=$2
tmpdir=$3
dir=$4
mkdir -p $dir $tmpdir $dir/phones

[ -f path.sh ] && . ./path.sh

utils/validate_dict_dir.pl $srcdir || exit 1;

# Create $tmpdir/lexicon.txt from $srcdir/lexicon.txt by
# adding the markers _B, _E, _S, _I depending on word position.
# In this recipe, these markers apply to silence also.

perl -ane '@A=split(" ",$_); $w = shift @A; @A>0||die;
  if(@A==1) { print "$w $A[0]_S\n"; } else { print "$w $A[0]_B ";
    for($n=1;$n<@A-1;$n++) { print "$A[$n]_I "; } print "$A[$n]_E\n"; } ' \
  <$srcdir/lexicon.txt >$tmpdir/lexicon.txt || exit 1;

# create $tmpdir/phone_map.txt
# this has the format (on each line)
# <original phone> <version 1 of original phone> <version 2 of original phone> ...
# where the different versions depend on word position.  For instance, we'd have
# AA AA_B AA_E AA_I AA_S
# and in the case of silence
# SIL SIL SIL_B SIL_E SIL_I SIL_S
# [because SIL on its own is one of the variants; this is for when it doesn't
#  occur inside a word but as an option in the lexicon.]

# This phone map expands the phone lists into all the word-position-dependent
# versions of the phone lists.

cat <(for x in `cat $srcdir/silence_phones.txt`; do for y in "" "" "_B" "_E" "_I" "_S"; do echo -n "$x$y "; done; echo; done) \
  <(for x in `cat $srcdir/nonsilence_phones.txt`; do for y in "" "_B" "_E" "_I" "_S"; do echo -n "$x$y "; done; echo; done) \
  > $tmpdir/phone_map.txt

mkdir -p $dir/phones # various sets of phones...

# Sets of phones for use in clustering, and making monophone systems.

if $share_silence_phones; then
  # build a roots file that will force all the silence phones to share the
  # same pdf's. [only the transitions will differ.]
  cat $srcdir/silence_phones.txt | awk '{printf("%s ", $0); } END{printf("\n");}' | cat - $srcdir/nonsilence_phones.txt | \
    utils/apply_map.pl $tmpdir/phone_map.txt > $dir/phones/sets.txt
  cat $dir/phones/sets.txt | awk '{if(NR==1) print "not-shared", "not-split", $0; else print "shared", "split", $0;}' > $dir/phones/roots.txt
else
  # different silence phones will have different GMMs.  [note: here, all "shared split" means
  # is that we may have one GMM for all the states, or we can split on states.  because they're
  # context-independent phones, they don't see the context.]
  cat $srcdir/{,non}silence_phones.txt | utils/apply_map.pl $tmpdir/phone_map.txt > $dir/phones/sets.txt
  cat $dir/phones/sets.txt | awk '{print "shared", "split", $0;}' > $dir/phones/roots.txt
fi

cat $srcdir/silence_phones.txt | utils/apply_map.pl $tmpdir/phone_map.txt | \
 awk '{for(n=1;n<=NF;n++) print $n;}' > $dir/phones/silence.txt
cat $srcdir/nonsilence_phones.txt | utils/apply_map.pl $tmpdir/phone_map.txt | \
 awk '{for(n=1;n<=NF;n++) print $n;}' > $dir/phones/nonsilence.txt
cp $srcdir/optional_silence.txt $dir/phones/optional_silence.txt
cp $dir/phones/silence.txt $dir/phones/context_indep.txt


cat $srcdir/extra_questions.txt | utils/apply_map.pl $tmpdir/phone_map.txt \
  >$dir/phones/extra_questions.txt

# Want extra questions about the word-start/word-end stuff. Make it separate for
# silence and non-silence.. probably doesn't really matter, as silence will rarely
# be inside a word.
for suffix in _B _E _I _S; do
 (for x in `cat $srcdir/nonsilence_phones.txt`; do echo -n "$x$suffix "; done; echo) >>$dir/phones/extra_questions.txt
done
for suffix in "" _B _E _I _S; do
 (for x in `cat $srcdir/silence_phones.txt`; do echo -n "$x$suffix "; done; echo) >>$dir/phones/extra_questions.txt
done


# add disambig symbols to the lexicon in $tmpdir/lexicon.txt
# and produce $tmpdir/lexicon_disambig.txt

ndisambig=`utils/add_lex_disambig.pl $tmpdir/lexicon.txt $tmpdir/lexicon_disambig.txt`
ndisambig=$[$ndisambig+1]; # add one disambig symbol for silence in lexicon FST.
echo $ndisambig > $tmpdir/lex_ndisambig

# Format of lexicon_disambig.txt:
# !SIL	SIL_S
# <SPOKEN_NOISE>	SPN_S #1
# <UNK>	SPN_S #2
# <NOISE>	NSN_S
# !EXCLAMATION-POINT	EH2_B K_I S_I K_I L_I AH0_I M_I EY1_I SH_I AH0_I N_I P_I OY2_I N_I T_E

( for n in `seq 0 $ndisambig`; do echo '#'$n; done ) >$dir/phones/disambig.txt

# Create phone symbol table.
echo "<eps>" | cat - $dir/phones/{silence,nonsilence,disambig}.txt | \
  awk '{n=NR-1; print $1, n;}' > $dir/phones.txt 

# Create a file that describes the word-boundary information for
# each phone.  5 categories.
cat $dir/phones/{silence,nonsilence}.txt | \
  awk '/_I$/{print $1, "internal"; next;} /_B$/{print $1, "begin"; next; }
      /_S$/{print $1, "singleton"; next;} /_E$/{print $1, "end"; next; }
      {print $1, "nonword";} ' > $dir/phones/word_boundary.txt

# Create word symbol table.
cat $tmpdir/lexicon.txt | awk '{print $1}' | sort | uniq  | \
 awk 'BEGIN{print "<eps> 0";} {printf("%s %d\n", $1, NR);} END{printf("#0 %d\n", NR+1);} ' \
  > $dir/words.txt || exit 1;

# format of $dir/words.txt:
#<eps> 0
#!EXCLAMATION-POINT 1
#!SIL 2
#"CLOSE-QUOTE 3
#...

silphone=`cat $srcdir/optional_silence.txt` || exit 1;

# Create the basic L.fst without disambiguation symbols, for use
# in training. 
utils/make_lexicon_fst.pl $tmpdir/lexicon.txt $sil_prob $silphone | \
  fstcompile --isymbols=$dir/phones.txt --osymbols=$dir/words.txt \
  --keep_isymbols=false --keep_osymbols=false | \
   fstarcsort --sort_type=olabel > $dir/L.fst || exit 1;

# The file oov.txt contains a word that we will map any OOVs to during
# training.
echo "$oov_word" > $dir/oov.txt || exit 1;
cat $dir/oov.txt | utils/sym2int.pl $dir/words.txt >$dir/oov.int # integer version of oov
# symbol, used in some scripts.



# Create these lists of phones in colon-separated integer list form too, 
# for purposes of being given to programs as command-line options.
for f in silence nonsilence optional_silence disambig context_indep; do
  utils/sym2int.pl $dir/phones.txt <$dir/phones/$f.txt >$dir/phones/$f.int
  utils/sym2int.pl $dir/phones.txt <$dir/phones/$f.txt | \
   awk '{printf(":%d", $1);} END{printf "\n"}' | sed s/:// > $dir/phones/$f.csl || exit 1;
done

for x in sets extra_questions; do
  utils/sym2int.pl $dir/phones.txt <$dir/phones/$x.txt > $dir/phones/$x.int || exit 1;
done

utils/sym2int.pl -f 3- $dir/phones.txt <$dir/phones/roots.txt \
   > $dir/phones/roots.int || exit 1;

utils/sym2int.pl -f 1 $dir/phones.txt <$dir/phones/word_boundary.txt \
  > $dir/phones/word_boundary.int || exit 1;


silphonelist=`cat $dir/phones/silence.csl`
nonsilphonelist=`cat $dir/phones/nonsilence.csl`
utils/gen_topo.pl $num_nonsil_states $num_sil_states $nonsilphonelist $silphonelist >$dir/topo


# Create the lexicon FST with disambiguation symbols, and put it in lang_test.
# There is an extra step where we create a loop to "pass through" the
# disambiguation symbols from G.fst.
phone_disambig_symbol=`grep \#0 $dir/phones.txt | awk '{print $2}'`
word_disambig_symbol=`grep \#0 $dir/words.txt | awk '{print $2}'`

utils/make_lexicon_fst.pl $tmpdir/lexicon_disambig.txt $sil_prob $silphone '#'$ndisambig | \
   fstcompile --isymbols=$dir/phones.txt --osymbols=$dir/words.txt \
   --keep_isymbols=false --keep_osymbols=false |   \
   fstaddselfloops  "echo $phone_disambig_symbol |" "echo $word_disambig_symbol |" | \
   fstarcsort --sort_type=olabel > $dir/L_disambig.fst || exit 1;
