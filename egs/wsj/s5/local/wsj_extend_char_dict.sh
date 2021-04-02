#!/usr/bin/env bash
# Copyright 2017   Hossein Hadian

# This script extends the word list by including OOVs from the training
# transcripts.
# Since no phonemes are involved, we need no G2P models/rules.
# In other words, this script is like wsj_extend_dict.sh except
# it deals with characters (i.e. graphemes) instead of phonemes
# so it's much simpler. Parts of this script are taken from
# EESEN (https://github.com/srvk/eesen)

if [ $# -ne 3 ]; then
  echo "usage: $0 <wsj-corpus-dir> <dict-src-dir> <dict-larger-dir>"
  echo "e.g.: $0 WSJ/13-32.1/ data/local/lang_char data/local/lang_char_larger"
  exit 1;
fi

if [ "`basename $1`" != 13-32.1 ]; then
  echo "Expecting the first argument to this script to end in 13-32.1"
  exit 1
fi

corpusdir=$1
srcdir=$2
dir=$3

mincount=2 # Minimum count of an OOV we include into the lexicon.

mkdir -p $dir
cp $srcdir/lexicon.txt $dir/lexicon.ori.txt
cp $srcdir/nonsilence_phones.txt $dir
cp $srcdir/silence_phones.txt $dir
cp $srcdir/optional_silence.txt $dir

# the original wordlist
cat $dir/lexicon.ori.txt | awk '{print $1}' | sort | uniq > $dir/wordlist.ori

# Get the training transcripts
echo "Getting the training transcripts, may take some time ..."

touch $dir/cleaned.gz
if [ `du -m $dir/cleaned.gz | cut -f 1` -eq 73 ]; then
  echo "Not getting cleaned data in $dir/cleaned.gz again [already exists]";
else
 gunzip -c $corpusdir/wsj1/doc/lng_modl/lm_train/np_data/{87,88,89}/*.z \
  | awk '/^</{next}{print toupper($0)}' | perl -e '
   open(F, "<$ARGV[0]")||die;
   while(<F>){ chop; $isword{$_} = 1; }
   while(<STDIN>) {
    @A = split(" ", $_);
    for ($n = 0; $n < @A; $n++) {
      $a = $A[$n];
      if (! $isword{$a} && $a =~ s/^([^\.]+)\.$/$1/) { # nonwords that end in "."
         # and have no other "." in them: treat as period.
         print "$a";
         if ($n+1 < @A) { print "\n"; }
      } else { print "$a "; }
    }
    print "\n";
  }
 ' $dir/wordlist.ori | gzip -c > $dir/cleaned.gz
fi

# Get unigram counts and the counts of the oov words
echo "Getting unigram counts"
gunzip -c $dir/cleaned.gz | tr -s ' ' '\n' | \
  awk '{count[$1]++} END{for (w in count) { print count[w], w; }}' | \
  sort -nr > $dir/unigrams

cat $dir/unigrams | awk -v dict=$dir/wordlist.ori \
  'BEGIN{while(getline<dict) seen[$1]=1;} {if(!seen[$2]){print;}}' \
   > $dir/oov.counts

echo "Most frequent unseen unigrams are: "
head $dir/oov.counts

# Select the OOVs whose counts > $mincount. Include these OOVs into the lexicon.
cat $dir/oov.counts | awk -v thresh=$mincount '{if ($1 >= thresh) { print $2; }}' > $dir/oovlist
cat $dir/oovlist | perl -e 'while(<>){ chop; $str="$_"; foreach $p (split("", $_)) {$str="$str $p"}; print "$str\n";}' > $dir/lexicon.oov.txt

# filter out oov words that have characters not in non-silence characters
cat $dir/lexicon.oov.txt | awk -v dict=$dir/nonsilence_phones.txt \
 'BEGIN{while(getline<dict) seen[$1]=1;} {for(i=2;i<=NF;i++) {if(!seen[$i]){break;}}; if (i==(NF+1)){print;}}' > $dir/lexicon.oov.filt.txt

# THe final expanded lexicon
cat $dir/lexicon.ori.txt $dir/lexicon.oov.filt.txt > $dir/lexicon.txt

echo "Number of OOVs we handled is `cat $dir/lexicon.oov.filt.txt | wc -l`"
echo "Created the larger lexicon $dir/lexicon.txt"
