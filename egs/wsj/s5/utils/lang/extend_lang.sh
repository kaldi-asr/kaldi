#!/usr/bin/env bash
# Copyright     2018  Johns Hopkins University (Author: Daniel Povey);
#               2019  Dongji Gao

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

# derived files, that go in data/lang/.

# Begin configuration section.
sil_prob=0.5
silprob_file=
# end configuration section

echo "$0 $@"  # Print the command line for logging

. utils/parse_options.sh

if [ $# -ne 3 ]; then
  echo "Usage: utils/extend_lang.sh <old-lang-dir> <lexicon> <new-lang-dir>"
  echo "e.g.: utils/extend_lang.sh data/lang data/local/dict_new_words/lexiconp.txt data/lang_new_words"
  echo ""
  echo "This script creates a lang/ directory <new-lang-dir> with L.fst and L_disambig.fst"
  echo "derived from the provided lexicon, but all other information being the same as the old"
  echo "lang/ directory, including the phones.txt and words.txt being compatible (however,"
  echo "words.txt may have new words, and phones.txt may have extra disambiguation symbols"
  echo "if needed).  We do not allow new phones."
  echo ""
  echo "CAUTION: the lexicon generated will only cover the words in the provided lexicon,"
  echo "which might not include all the words in words.txt.  You should make sure your"
  echo "lexicon is a superset of the original lexicon used to generate <old-lang-dir>,"
  echo "if this would be a problem for your scenario."
  echo ""
  echo "The basename of <lexicon> must be either lexicon.txt, lexiconp.txt or lexiconp_silprob.txt."
  echo ""
  echo "Options"
  echo "     --sil-prob <probability of silence>             # default: 0.5 [must have 0 <= silprob < 1]"
  echo "     --silprob-file <file contains silence probability>    # must be provided if lexicon is lexiconp_silprob.txt"
  exit 1;
fi

srcdir=$1
lexicon=$2
dir=$3

[ -f path.sh ] && . ./path.sh


for f in $srcdir/phones.txt $lexicon; do
  if [ ! -f $f ]; then
    echo "$0: expected file $f to exist"
    exit 1
  fi
done

if ! awk '{if(NF < 2) exit(1)} END{if(NR==0) exit(1)}' <$lexicon; then
  echo "$0: it looks like there words without pronunciations or.."
  echo "  ...blank lines in $lexicon, or it is empty."
  exit 1
fi

mkdir -p $dir

if [ -d $dir/phones ]; then rm -r $dir/phones; fi

cp -r $srcdir/phones $dir/

for f in oov.int oov.txt phones.txt topo words.txt; do
  cp $srcdir/$f $dir/
done

tmpdir=$dir/temp
rm -r $tmpdir 2>/dev/null
mkdir -p $tmpdir

silprob=false

if [ $(basename $lexicon) == "lexiconp_silprob.txt" ]; then
  silprob=true
  if [ -z $silprob_file ] ; then
    echo "silprob_file not provided, checking $srcdir"
    if [ -f $srcdir/silprob.txt ]; then
        silprob_file=$srcdir/silprob.txt
        echo "silprob_file found in $srcdir"
    else
        echo "silprob_file not found in $srcdir" && exit 1;
    fi
  else
    if [ ! -f $silprob_file ]; then
      echo "$silprob_file does not exist" && exit 1;
    fi
  fi
elif [ $(basename $lexicon) != lexiconp.txt ]; then
  echo "$0: currently this script only supports the lexiconp.txt or lexiconp_silprob.txt format;"
  echo " ... your lexicon has to have that filename."
fi

# Get the list of extra words.
awk -v w=$srcdir/words.txt 'BEGIN{while(getline <w) seen[$1] = $1} { if (!($1 in seen)) oov[$1] = 1}
                     END{ for(k in oov) print k;}' <$lexicon >$tmpdir/extra_words.txt

# Add entries to words.txt for all the words that were not previously in the
# lexicon.
highest_number=$(tail -n 1 $srcdir/words.txt | awk '{print $2}')
awk -v start=$highest_number '{print $1, NR+start}' <$tmpdir/extra_words.txt >>$dir/words.txt
echo "$0: added $(wc -l <$tmpdir/extra_words.txt) extra words to words.txt"

if [ -f $dir/phones/nonterminals.txt ]; then
  # extra grammar-decoding-related options for getting the lexicon.
  grammar_opts="--left-context-phones=$dir/phones/left_context_phones.txt --nonterminals=$srcdir/phones/nonterminals.txt"
else
  grammar_opts=""
fi

if [ -f $dir/phones/word_boundary.txt ]; then
  # was `if $position_dependent_phones; then..` in prepare_lang.sh
  if "$silprob"; then
    perl -ane '@A=split(" ",$_); $w = shift @A; $p = shift @A; $silword_p = shift @A;
              $wordsil_f = shift @A; $wordnonsil_f = shift @A; @A>0||die;
         if(@A==1) { print "$w $p $silword_p $wordsil_f $wordnonsil_f $A[0]_S\n"; }
         else { print "$w $p $silword_p $wordsil_f $wordnonsil_f $A[0]_B ";
         for($n=1;$n<@A-1;$n++) { print "$A[$n]_I "; } print "$A[$n]_E\n"; } ' \
         < $lexicon > $tmpdir/lexiconp_silprob.txt
  else
    perl -ane '@A=split(" ",$_); $w = shift @A; $p = shift @A; @A>0||die;
           if(@A==1) { print "$w $p $A[0]_S\n"; } else { print "$w $p $A[0]_B ";
           for($n=1;$n<@A-1;$n++) { print "$A[$n]_I "; } print "$A[$n]_E\n"; } ' \
         < $lexicon > $tmpdir/lexiconp.txt || exit 1;
  fi
else
  if "$silprob"; then
    cp $lexicon $tempdir/lexiconp_silprob.txt
  else
    cp $lexicon $tmpdir/lexiconp.txt
  fi
fi

# Check that there are no unseen phones in the lexicon.
if "$silprob"; then
  if ! utils/sym2int.pl -f 6- $srcdir/phones.txt $tmpdir/lexiconp_silprob.txt >/dev/null; then
    echo "$0: it looks like there are unseen phones in your lexicon $lexicon"
    exit 1
  fi
else 
  if ! utils/sym2int.pl -f 3- $srcdir/phones.txt $tmpdir/lexiconp.txt >/dev/null; then
    echo "$0: it looks like there are unseen phones in your lexicon $lexicon"
    exit 1
  fi
fi

if "$silprob"; then
  ndisambig=$(utils/add_lex_disambig.pl --pron-probs --sil-probs $tmpdir/lexiconp_silprob.txt $tmpdir/lexiconp_silprob_disambig.txt)
else
  ndisambig=$(utils/add_lex_disambig.pl --pron-probs $tmpdir/lexiconp.txt $tmpdir/lexiconp_disambig.txt)
fi

ndisambig=$[ndisambig+1]  # Add one to disambiguate silence.

# we'll need to figure out whether any of these disambiguation symbols are
# absent from our current disambiguation phones.. if they are, then we need to
# add them as new disambiguation symbols to phones.txt.
for n in $(seq 0 $ndisambig); do
  sym='#'$n; if ! grep -w -q "$sym" $dir/phones/disambig.txt; then echo "$sym"; fi
done > $tmpdir/extra_disambig.txt
highest_number=$(tail -n 1 $srcdir/phones.txt | awk '{print $2}')
awk -v start=$highest_number '{print $1, NR+start}' <$tmpdir/extra_disambig.txt >>$dir/phones.txt
echo "$0: added $(wc -l <$tmpdir/extra_disambig.txt) extra disambiguation symbols to phones.txt"

# add extra_disambig symbols into disambig.txt
cat $tmpdir/extra_disambig.txt >> $dir/phones/disambig.txt
utils/sym2int.pl $dir/phones.txt <$dir/phones/disambig.txt >$dir/phones/disambig.int
utils/sym2int.pl $dir/phones.txt <$dir/phones/disambig.txt | \
  awk '{printf(":%d", $1);} END{printf "\n"}' | sed s/:// > $dir/phones/disambig.csl

silphone=`cat $srcdir/phones/optional_silence.txt` || exit 1;
[ -z "$silphone" ] && \
  ( echo "You have no optional-silence phone; it is required in the current scripts"
    echo "but you may use the option --sil-prob 0.0 to stop it being used." ) && \
   exit 1;

if "$silprob"; then
  # remove the silprob
  cat $tmpdir/lexiconp_silprob.txt |\
    awk '{
      for(i=1; i<=NF; i++) {
        if(i!=3 && i!=4 && i!=5) printf("%s\t", $i); if(i==NF) print "";
      }
    }' > $tmpdir/lexiconp.txt
fi

# First remove pron-probs from the lexicon.
perl -ape 's/(\S+\s+)\S+\s+(.+)/$1$2/;' <$tmpdir/lexiconp.txt >$tmpdir/align_lexicon.txt

# Note: here, $silphone will have no suffix e.g. _S because it occurs as optional-silence,
# and is not part of a word.
[ ! -z "$silphone" ] && echo "<eps> $silphone" >> $tmpdir/align_lexicon.txt

cat $tmpdir/align_lexicon.txt | \
  perl -ane '@A = split; print $A[0], " ", join(" ", @A), "\n";' | sort | uniq > $dir/phones/align_lexicon.txt

if [ -f $dir/phones/nonterminals.txt ]; then
  for w in "#nonterm_begin" "#nonterm_end" $(cat $dir/phones/nonterminals.txt); do
    echo $w $w  # These are words without pronunciations, so leave those prons
                # empty.
    done >> $dir/phones/align_lexicon.txt
fi

# create phones/align_lexicon.int from phones/align_lexicon.txt
cat $dir/phones/align_lexicon.txt | utils/sym2int.pl -f 3- $dir/phones.txt | \
  utils/sym2int.pl -f 1-2 $dir/words.txt > $dir/phones/align_lexicon.int

# Create the basic L.fst without disambiguation symbols, for use
# in training.
if "$silprob"; then
  utils/lang/make_lexicon_fst_silprob.py $grammar_opts --sil-phone=$silphone \
         $tmpdir/lexiconp_silprob.txt $silprob_file | \
      fstcompile --isymbols=$dir/phones.txt --osymbols=$dir/words.txt \
        --keep_isymbols=false --keep_osymbols=false |   \
      fstarcsort --sort_type=olabel > $dir/L.fst || exit 1;
else
  utils/lang/make_lexicon_fst.py $grammar_opts --sil-prob=$sil_prob --sil-phone=$silphone \
           $tmpdir/lexiconp.txt | \
      fstcompile --isymbols=$dir/phones.txt --osymbols=$dir/words.txt \
        --keep_isymbols=false --keep_osymbols=false | \
      fstarcsort --sort_type=olabel > $dir/L.fst || exit 1;
fi


# and create the version that has disambiguation symbols.
if "$silprob"; then
  utils/lang/make_lexicon_fst_silprob.py $grammar_opts \
    --sil-phone=$silphone --sil-disambig='#'$ndisambig \
    $tmpdir/lexiconp_silprob_disambig.txt $silprob_file | \
    fstcompile --isymbols=$dir/phones.txt --osymbols=$dir/words.txt \
      --keep_isymbols=false --keep_osymbols=false |   \
    fstaddselfloops  $dir/phones/wdisambig_phones.int $dir/phones/wdisambig_words.int | \
    fstarcsort --sort_type=olabel > $dir/L_disambig.fst || exit 1;
else
  utils/lang/make_lexicon_fst.py $grammar_opts \
    --sil-prob=$sil_prob --sil-phone=$silphone --sil-disambig='#'$ndisambig \
       $tmpdir/lexiconp_disambig.txt | \
     fstcompile --isymbols=$dir/phones.txt --osymbols=$dir/words.txt \
                --keep_isymbols=false --keep_osymbols=false | \
     fstaddselfloops $dir/phones/wdisambig_phones.int $dir/phones/wdisambig_words.int | \
     fstarcsort --sort_type=olabel > $dir/L_disambig.fst || exit 1;
fi


echo "$(basename $0): validating output directory"
# the --skip-generate-words-check option is needed because L.fst may not actually
# contain all the words in words.txt.
! utils/validate_lang.pl --skip-generate-words-check $dir && echo "$(basename $0): error validating output" &&  exit 1;

exit 0;
