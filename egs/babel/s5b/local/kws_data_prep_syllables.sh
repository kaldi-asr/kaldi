#!/usr/bin/env bash

# Copyright 2012  Johns Hopkins University (Author: Guoguo Chen)
# Apache 2.0.

# Begin configuration section.  
silence_word=  # Optional silence word to insert (once) between words of the transcript.
# End configuration section.

echo $0 "$@"

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;


if [ $# -ne 4 ]; then
   echo "Usage: local/kws_data_prep_syllables.sh [options] <lang-dir> <data-dir> <syllable-lexicon> <kws-data-dir>"
   echo " e.g.: local/kws_data_prep_syllables.sh data/lang/ data/dev10h/  SIL data/kws/"
   echo "Input is in <kws-data-dir>: kwlist.xml, ecf.xml (rttm file not needed)."
   echo "The lang directory is expected to be syllable-level.  The syllable-lexicon "
   echo "is a text file with lines of the form:"
   echo "word  syllable1 syllable2"
   echo "This script is as kws_data_prep.sh, except that the output keywords.fsts"
   echo "contains the various alternative syllable-level pronunciations of the input"
   echo "words."
   echo "Output is in <kws-data/dir>: keywords.txt, kwlist_invocab.xml,"
   echo "    kwlist_outvocab.xml, keywords.fsts; note that the only syllable-level"
   echo "  output (and the only one that really matters) is keywords.fsts"
   echo "Note: most important output is keywords.fsts"
   echo " Options:"
   echo "  --silence-word   <silence-word>        # Note, this is required.  It is a word, e.g. SIL,"
   echo "                                         # in the syllable lexicon, that's optional."
   exit 1;
fi

langdir=$1;
datadir=$2;
syllable_lexicon=$3
kwsdatadir=$4
keywords=$kwsdatadir/kwlist.xml

[ -z $silence_word ] && echo "--silence-word option is required" && exit 1;

mkdir -p $kwsdatadir;

cat $keywords | perl -e '
  #binmode STDIN, ":utf8"; 
  binmode STDOUT, ":utf8"; 

  use XML::Simple;
  use Data::Dumper;

  my $data = XMLin(\*STDIN);

  #print Dumper($data->{kw});
  foreach $kwentry (@{$data->{kw}}) {
    #print Dumper($kwentry);
    print "$kwentry->{kwid}\t$kwentry->{kwtext}\n";
  }
' > $kwsdatadir/keywords.txt

[ ! -s "$syllable_lexicon" ] && echo "No such file '$syllable_lexicon' (syllable lexicon), or empty file." && exit 1;

# The word symbols on the first entry of $syllable_lexicon will be given a symbol-table
# file.  We just use this symbol table in this script; the values will never appear
# elsewhere.  

mkdir -p $kwsdatadir/temp

# Remove any lines with symbols we don't have in our symbol vocabulary.
temp_syllable_lexicon=$kwsdatadir/temp/syllable_lexicon.in
cat $syllable_lexicon | sym2int.pl --map-oov 123456789 -f 2- $langdir/words.txt | grep -v -w 123456789 | \
  int2sym.pl -f 2- $langdir/words.txt > $temp_syllable_lexicon

n1=`cat $syllable_lexicon | wc -l`
n2=`cat $temp_syllable_lexicon | wc -l`
echo "After removing OOV symbols from word-to-syllable lexicon, #lines changed from $n1 to $n2"


if $case_insensitive; then
  echo "Running case insensitive processing"
  # we turn the first element of each line of $temp_syllable_lexicon into upper case.
  tr '[:lower:]' '[:upper:]' < $temp_syllable_lexicon | awk '{print $1}' | \
     paste - <(awk '{for(n=2;n<=NF;n++) { printf("%s ", $n); } print ""; }'  <$temp_syllable_lexicon) \
    > $kwsdatadir/temp/syllable_lexicon.txt || exit 1;

  # We turn all but the first element of each line in $kwsdatadir/keywords.txt
  # into upper case.
  tr '[:lower:]' '[:upper:]' < $kwsdatadir/keywords.txt | \
     awk '{for(n=2;n<=NF;n++) { printf("%s ", $n); } print ""; }' | \
     paste <(awk '{print $1}'  <$kwsdatadir/keywords.txt) - \
    > $kwsdatadir/temp/keywords.txt || exit 1;
else
  cp $temp_syllable_lexicon $kwsdatadir/temp/syllable_lexicon.txt || exit 1;
  cp $kwsdatadir/keywords.txt $kwsdatadir/temp/ || exit 1;
fi

cat $kwsdatadir/temp/syllable_lexicon.txt | awk '{print $1}' | sort | uniq | \
 awk 'BEGIN{print "<eps> 0";} {print $1, NR;}' > $kwsdatadir/temp/words.txt

sym2int.pl --map-oov 0 -f 2- $kwsdatadir/temp/words.txt < $kwsdatadir/temp/keywords.txt \
  > $kwsdatadir/temp/keywords_all.int

cat $kwsdatadir/temp/keywords_all.int | \
  grep -v " 0 " | grep -v " 0$" > $kwsdatadir/keywords.int

cut -f 1 -d ' ' $kwsdatadir/keywords.int | \
  local/subset_kwslist.pl $keywords > $kwsdatadir/kwlist_invocab.xml

cat $kwsdatadir/temp/keywords_all.int | \
  egrep " 0 | 0$" | cut -f 1 -d ' ' | \
  local/subset_kwslist.pl $keywords > $kwsdatadir/kwlist_outvocab.xml

local/make_lexicon_fst_special.pl $kwsdatadir/temp/syllable_lexicon.txt $silence_word | \
  sym2int.pl -f 4 $kwsdatadir/temp/words.txt | \
  sym2int.pl -f 3 $langdir/words.txt | \
  fstcompile | \
  fstarcsort --sort_type=olabel > $kwsdatadir/temp/L.fst || exit 1;

# Compile keywords into FSTs, compose with lexicon to get syllables
#  and project on the input (keeping only syllable labels), 
# before writing to keywords.fsts

transcripts-to-fsts ark:$kwsdatadir/keywords.int ark:- | \
  fsttablecompose $kwsdatadir/temp/L.fst ark:- ark,t:- | \
  awk '{if (NF < 4) { print; } else { print $1, $2, $3, $3, $5; }}' > \
  $kwsdatadir/keywords.fsts

# Create utterance id for each utterance
cat $datadir/segments | \
  awk '{print $1}' | \
  sort | uniq | perl -e '
  $idx=1;
  while(<>) {
    chomp;
    print "$_ $idx\n";
    $idx++;
  }' > $kwsdatadir/utter_id

# Map utterance to the names that will appear in the rttm file. You have 
# to modify the commands below accoring to your rttm file
cat $datadir/segments | awk '{print $1" "$2}' | sort | uniq > $kwsdatadir/utter_map;

echo "Kws data preparation succeeded"
