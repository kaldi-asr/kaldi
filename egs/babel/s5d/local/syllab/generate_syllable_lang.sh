#!/usr/bin/env bash
# Copyright (c) 2015, Johns Hopkins University (Yenda Trmal <jtrmal@gmail.com>)
# License: Apache 2.0

# Begin configuration section.
cmd=run.pl
# End configuration section
. ./utils/parse_options.sh
. ./path.sh



set -e -o pipefail
set -o nounset                              # Treat unset variables as an error

data=$1
llang=$2
lang=$3
out=$4
lout=$5

test -d $lout && rm -rf $lout
mkdir -p $lout
test -d $out && rm -rf $out
cp -R $lang $out
rm -rf $out/tmp $out/L.fst $out/L_disambig.fst $out/G.fst $out/words.txt
rm -rf $out/phones/word_boundary.{int,txt}

echo "Generating lexicons.."
if [ -f $lang/phones/word_boundary.int ] ; then
  echo "Position dependent phones system..."
  if [ -f $llang/lexiconp.txt ] ; then
    echo "Using probabilistic lexicon..."
    cat $llang/lexiconp.txt | local/syllab/create_syllables.pl --with-probs\
      $lout/lex.syllabs2phones.txt $lout/lex.words2syllabs.txt $lout/lex.words2phones.txt
  else
    echo "Using plain lexicon..."
    cat $llang/lexicon.txt | local/syllab/create_syllables.pl \
      $lout/lex.syllabs2phones.txt $lout/lex.words2syllabs.txt $lout/lex.words2phones.txt
  fi
else
  echo "Position independent phones system..."
  if [ -f $llang/lexiconp.txt ] ; then
    echo "Using probabilistic lexicon..."
    cat $llang/lexiconp.txt | local/syllab/create_syllables.pl --with-probs --position-independent-phones\
      $lout/lex.syllabs2phones.txt $lout/lex.words2syllabs.txt $lout/lex.words2phones.txt
  else
    echo "Using plain lexicon..."
    cat $llang/lexicon.txt | local/syllab/create_syllables.pl --position_independent_phones\
      $lout/lex.syllabs2phones.txt $lout/lex.words2syllabs.txt $lout/lex.words2phones.txt
  fi
fi
cp $lout/lex.{syllabs2phones,words2syllabs,words2phones}.txt $out

#We will fake the words.txt file
(
  echo "<eps>";
  cut -f 1 $out/lex.syllabs2phones.txt;
  echo -e "#0\n<s>\n</s>";
) | nl -v 0 | awk '{print $2, $1}' > $out/syllabs.txt
ln -s syllabs.txt $out/words.txt
cp $lang/words.txt $out/real_words.txt


#Figure out the "OOV" token
oovword=$(cat $lang/oov.txt)
oovsyl=$(grep -w -F "$oovword" $out/lex.words2syllabs.txt | \
        awk '{if (NF == 2) { print $2;}
        else {print "Error, oov word has more than one syllable "; exit 1;}}')

echo $oovsyl > $out/oov.txt
grep -w -F "$oovsyl" $out/words.txt | awk '{print $2}' > $out/oov.int

phone_disambig_symbol=$(grep '#0' $out/phones.txt | awk '{print $2}')
word_disambig_symbol=$(grep '#0' $out/words.txt | awk '{print $2}')

optional_sil=$(cat $out/phones/optional_silence.txt)
utils/add_lex_disambig.pl  $out/lex.syllabs2phones.txt $out/lex.syllabs2phones.disambig.txt > /dev/null
cat $out/lex.syllabs2phones.disambig.txt | sort -u > $lout/lexicon.txt

if [ -f $out/phones/wdisambig_words.int  ]; then
  echo $word_disambig_symbol > $out/phones/wdisambig_words.int
fi

echo "<eps> SIL" | cat - $lout/lexicon.txt | perl -ane 'print $F[0], " ", join(" ", @F), "\n";' | \
  sed 's/ #[0-9]$//g' > $out/phones/align_lexicon.txt
cat $lout/lexicon.txt | perl -ane 'print $F[0], "\t1.0\t", join(" ", @F[1..$#F]), "\n";' \
   > $lout/lexiconp.txt

cat $out/phones/align_lexicon.txt |\
  sym2int.pl -f 3- $out/phones.txt |\
  sym2int.pl -f 1-2 $out/words.txt \
  > $out/phones/align_lexicon.int

ndisambig=$(cat $out/phones/disambig.int | wc -l)
ndisambig=$[$ndisambig-1]


#Compile the lexicons
echo "Compiling words2syllables FST"
utils/make_lexicon_fst.pl $out/lex.words2syllabs.txt | \
  fstcompile --isymbols=$out/syllabs.txt --osymbols=$lang/words.txt \
    --keep_isymbols=false --keep_osymbols=false| \
  fstarcsort --sort_type=olabel > $out/lex.words2syllabs.fst

echo "Compiling L.fst and L_disambig.fst"
sil=$(cat $lang/phones/optional_silence.txt)
utils/make_lexicon_fst.pl $out/lex.syllabs2phones.txt 0.5 $sil | \
  fstcompile --isymbols=$lang/phones.txt --osymbols=$out/syllabs.txt \
    --keep_isymbols=false --keep_osymbols=false| \
  fstarcsort --sort_type=olabel > $out/lex.syllabs2phones.fst
ln -s lex.syllabs2phones.fst $out/L.fst


utils/make_lexicon_fst.pl $out/lex.syllabs2phones.disambig.txt 0.5 $sil '#'$ndisambig | \
  fstcompile --isymbols=$lang/phones.txt --osymbols=$out/syllabs.txt \
    --keep_isymbols=false --keep_osymbols=false| \
  fstaddselfloops  "echo $phone_disambig_symbol |" "echo $word_disambig_symbol |"|\
  fstarcsort --sort_type=olabel > $out/lex.syllabs2phones.disambig.fst
ln -s lex.syllabs2phones.disambig.fst $out/L_disambig.fst

echo "Validating the output lang dir"
utils/validate_lang.pl $out || exit 1

perl -i -pe 's/#1$//g' $lout/lexicon.txt $lout/lexiconp.txt

echo "Done OK."
exit 0
