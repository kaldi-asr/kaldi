#!/bin/bash
# Copyright Daniel Povey  2012
# Apache 2.0

# This script prepares a directory such as data/lang/, in the standard format,
# given a source directory containing a dictionary lexicon.txt in a form like:
# word phone1 phone2 ... phonen
# per line (alternate prons would be separate lines).
# and also files silence_phones.list and nonsilence_phones.list
# This script adds word-position-dependent phones and constructs a host of other
# derived files, that go in data/lang/.

if [ $# -ne 3 ]; then 
  echo "usage: utils/prepare_lang.sh <dict-src-dir> <tmp-dir> <lang-dir>"
  echo "e.g.: utils/prepare_lang.sh data/local/dict data/local/lang data/lang"
  exit 1;
fi


tmpdir=$2
srcdir=$1
dir=$3
mkdir -p $dir $tmpdir $dir/phones

# Create $tmpdir/lexicon.txt from $srcdir/lexicon.txt by
# adding the markers _B, _E, _S, _I depending on word position.
# In this recipe, these markers apply to silence also.

perl -ane '@A=split(" ",$_); $w = shift @A; @A>0||die;
  if(@A==1) { print "$w $A[0]_S\n"; } else { print "$w $A[0]_B ";
    for($n=1;$n<@A-1;$n++) { print "$A[$n]_I "; } print "$A[$n]_E\n"; } ' \
  <$srcdir/lexicon.txt >$tmpdir/lexicon.txt || exit 1;

for f in $srcdir/{,non}silence_phones.list; do
  [ ! -f $f ] && echo "No such file $f" && exit 1;
done

echo "<eps>" | \
 cat - <(for x in `cat $srcdir/silence_phones.list`; do for y in "" "_B" "_E" "_I" "_S"; do echo "$x$y"; done; done) | \
 cat - <(for x in `cat $srcdir/nonsilence_phones.list`; do for y in "_B" "_E" "_I" "_S"; do echo "$x$y"; done; done) | \
 awk '{ n=NR-1; print $1, n; }' > $tmpdir/phones_nodisambig.txt 

# Now create lists of silence phones and nonsilence phones; and word-begin and word-end
# information.
rm $tmpdir/{,non}silence_phones.list 2>/dev/null
for x in `grep -v -w '<eps>' $tmpdir/phones_nodisambig.txt | awk '{print $1}'`; do  
  basephone=`echo $x | sed s/_[BEIS]$//`;
  if grep -w $basephone <$srcdir/silence_phones.list >/dev/null; then # was silence
    echo $x >>$tmpdir/silence_phones.list
  else
    echo $x >>$tmpdir/nonsilence_phones.list
  fi
done
  
# (0), this is more data-preparation than data-formatting;
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


# Create phones file with disambiguation symbols.
utils/add_disambig.pl --include-zero $tmpdir/phones_nodisambig.txt \
  `cat $tmpdir/lex_ndisambig` > $dir/phones.txt

# Create 3 subsets of the phones: silence, nonsilence, and disambig.
cp $tmpdir/silence_phones.list $dir/phones/silence.txt
cp $dir/phones/silence.txt $dir/phones/context_indep.txt # context-independent phones.
 # In general the silence phones and the context-independent phones will be the
 # same set (this is specified in the roots.txt file created below).
cp $tmpdir/nonsilence_phones.list $dir/phones/nonsilence.txt
grep -E '^#[0-9]+' $dir/phones.txt | awk '{print $1}' > $dir/phones/disambig.txt

# Create these lists of phones in colon-separated integer list form too, 
# for purposes of being given to programs as command-line options.
for f in silence nonsilence disambig context_indep; do
  utils/sym2int.pl $dir/phones.txt <$dir/phones/$f.txt | \
   awk '{printf(":%d", $1);} END{printf "\n"}' | sed s/:// > $dir/phones/$f.csl || exit 1;
done

# Create a file that describes the word-boundary information for
# each phone.  5 categories.
mkdir -p $dir/phones
grep -v -w '<eps>' $tmpdir/phones_nodisambig.txt | awk '{print $1;}' | \
  awk '/_I$/{print $1, "internal"; next;} /_B$/{print $1, "begin"; next; }
       /_S$/{print $1, "singleton"; next;} /_E$/{print $1, "end"; next; }
       {print $1, "nonword";} ' > $dir/phones/word_boundary.txt



cat $tmpdir/lexicon.txt | awk '{print $1}' | sort | uniq  | \
 awk 'BEGIN{print "<eps> 0";} {printf("%s %d\n", $1, NR);} END{printf("#0 %d\n", NR+1);} ' \
  > $dir/words.txt || exit 1;

# format of $dir/words.txt:
#<eps> 0
#!EXCLAMATION-POINT 1
#!SIL 2
#"CLOSE-QUOTE 3
#...

# Create the basic L.fst without disambiguation symbols, for use
# in training. 
utils/make_lexicon_fst.pl $tmpdir/lexicon.txt 0.5 SIL | \
  fstcompile --isymbols=$dir/phones.txt --osymbols=$dir/words.txt \
  --keep_isymbols=false --keep_osymbols=false | \
   fstarcsort --sort_type=olabel > $dir/L.fst || exit 1;

# The file oov.txt contains a word that we will map any OOVs to during
# training.
echo "<SPOKEN_NOISE>" > $dir/oov.txt || exit 1;
cat $dir/oov.txt | utils/sym2int.pl $dir/words.txt >$dir/oov.int # integer version of oov
# symbol, used in some scripts.

# (2)
# Create phonesets_*.txt and extra_questions.txt ...
# phonesets_mono.txt is sets of phones that are shared when building the monophone system
# and when asking questions based on an automatic clustering of phones, for the
# triphone system.  extra_questions.txt is some pre-defined extra questions about
# position and stress that split apart the categories we created in phonesets.txt.
# in extra_questions.txt there is also a question about silence phones, since we 
# don't include them in our automatically generated clustering of phones.

mkdir -p $dir/phones

cat $dir/phones/silence.txt | awk '{printf("%s ", $1);} END{printf "\n";}' \
  > $dir/phones/sets_mono.txt || exit 1;

cat $dir/phones/nonsilence.txt | \
  perl -e 'while(<>){ m:([A-Za-z]+)(\d*)(_.)?: || die "Bad line $_"; 
     $phone=$1; $stress=$2; $position=$3;
     if($phone eq $curphone){ print " $phone$stress$position"; }
  else { if(defined $curphone){ print "\n"; } $curphone=$phone;  print "$phone$stress$position";  }} print "\n"; ' \
 >> $dir/phones/sets_mono.txt || exit 1;

grep -v -w `head -1 $dir/phones/silence.txt` $dir/phones/sets_mono.txt \
  > $dir/phones/sets_cluster.txt || exit 1;

cat $dir/phones/silence.txt | awk '{printf("%s ", $1);} END{printf "\n";}' \
  > $dir/phones/extra_questions.txt
cat $dir/phones/nonsilence.txt | perl -e 'while(<>){ m:([A-Za-z]+)(\d*)(_.)?: || die "Bad line $_"; 
     $phone=$1; $stress=$2; $pos=$3;
     $full_phone ="$1$2$3";
     $pos2list{$pos} = $pos2list{$pos} .  $full_phone . " ";
     $stress2list{$stress} = $stress2list{$stress} .  $full_phone . " ";
   } 
   foreach $k (keys %pos2list) { print "$pos2list{$k}\n"; } 
   foreach $k (keys %stress2list) { print "$stress2list{$k}\n"; }  ' \
 >> $dir/phones/extra_questions.txt || exit 1;


( # Creating the "roots file" for building the context-dependent systems...
  # we share the roots across all the versions of each real phone.  We also
  # share the states of the 3 forms of silence.  "not-shared" here means the
  # states are distinct p.d.f.'s... normally we would automatically split on
  # the HMM-state but we're not making silences context dependent.
  cat $dir/phones/silence.txt | \
    awk 'BEGIN {printf("not-shared not-split ");} {printf("%s ",$1);} END{printf "\n";}';
  cat $dir/phones/nonsilence.txt | \
    perl -e 'while(<>){ m:([A-Za-z]+)(\d*)(_.)?: || die "Bad line $_"; 
            $phone=$1; $stress=$2; $position=$3;
      if($phone eq $curphone){ print " $phone$stress$position"; }
      else { if(defined $curphone){ print "\n"; } $curphone=$phone; 
            print "shared split $phone$stress$position";  }} print "\n"; '
 ) > $dir/phones/roots.txt || exit 1;

for x in sets_mono sets_cluster extra_questions disambig; do
  utils/sym2int.pl $dir/phones.txt <$dir/phones/$x.txt > $dir/phones/$x.int || exit 1;
done

utils/sym2int.pl -f 3- $dir/phones.txt <$dir/phones/roots.txt \
   > $dir/phones/roots.int || exit 1;

utils/sym2int.pl -f 1 $dir/phones.txt <$dir/phones/word_boundary.txt \
  > $dir/phones/word_boundary.int || exit 1;



silphonelist=`cat $dir/phones/silence.csl | sed 's/:/ /g'`
nonsilphonelist=`cat $dir/phones/nonsilence.csl | sed 's/:/ /g'`
cat conf/topo.proto | sed "s:NONSILENCEPHONES:$nonsilphonelist:" | \
   sed "s:SILENCEPHONES:$silphonelist:" > $dir/topo


# Create the lexicon FST with disambiguation symbols, and put it in lang_test.
# There is an extra step where we create a loop to "pass through" the
# disambiguation symbols from G.fst.
phone_disambig_symbol=`grep \#0 $dir/phones.txt | awk '{print $2}'`
word_disambig_symbol=`grep \#0 $dir/words.txt | awk '{print $2}'`

utils/make_lexicon_fst.pl $tmpdir/lexicon_disambig.txt 0.5 SIL '#'$ndisambig | \
   fstcompile --isymbols=$dir/phones.txt --osymbols=$dir/words.txt \
   --keep_isymbols=false --keep_osymbols=false |   \
   fstaddselfloops  "echo $phone_disambig_symbol |" "echo $word_disambig_symbol |" | \
   fstarcsort --sort_type=olabel > $dir/L_disambig.fst || exit 1;

