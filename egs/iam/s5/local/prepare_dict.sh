#!/usr/bin/env bash

#Copyright      2017  Chun Chieh Chang
#               2017  Ashish Arora

# This module prepares dictionary directory. It creates lexicon.txt,
#    silence_phones.txt, optional_silence.txt and extra_questions.txt.
#
#    Eg. local/prepare_dict.sh data/train/ data/train/dict


train_text=$1
test_text=$2
dir=$3

mkdir -p $dir

# reads the data from text files and write all unique words in lexicon.txt file
# find all unique words
cat $train_text/text | awk '{ for(i=2;i<=NF;i++) print $i;}' | sort -u >train_words

# write words in following format: Ben B e n
awk '{
  printf("%s", $1);
    for(j=1;j<=length($1);++j) {
      printf(" %s", substr($1, j, 1));
    }
  printf("\n");
}' "train_words" | sort -k1 > lexicon_with_hash

#replace '#' with '<HASH>'
sed 's/\#/<HASH>/2' lexicon_with_hash > $dir/lexicon.txt
rm -rf lexicon_with_hash train_words

cut -d' ' -f2- $dir/lexicon.txt | tr ' ' '\n' | sort -u >$dir/nonsilence_phones.txt || exit 1;

( echo '<sil> SIL'; ) >> $dir/lexicon.txt || exit 1;
( echo '<unk> SIL'; ) >> $dir/lexicon.txt || exit 1;

( echo SIL ) > $dir/silence_phones.txt

echo SIL >$dir/optional_silence.txt

echo -n "" >$dir/extra_questions.txt
