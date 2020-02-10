#!/usr/bin/env bash

#adapted from fisher dict preparation script, Author: Pawel Swietojanski
# Copyright 2019  IBM Corp. (Author: Michael Picheny) Adapted AMI recipe to MALACH corpus


if [ $# -ne 1 ]; then
  echo "Usage: $0 <malach-dir>"
  echo " <malach-dir> is download space."
  exit 1;
fi

malach_dir=$1

dir=data/local/dict
mkdir -p $dir
mkdir -p $dir/Malachdict
echo "Getting Malach dictionary"
cp  $malach_dir/lexicon.txt  $dir/Malachdict

# Copy over Malach files from the distribution
cp $malach_dir/silence_phones.txt $dir/silence_phones.txt
cp $malach_dir/optional_silence.txt $dir/optional_silence.txt
cp $malach_dir/nonsilence_phones.txt $dir/nonsilence_phones.txt
cp $malach_dir/nonsilence_phones.txt $dir/nonsilence_phones.txt
cp $malach_dir/lexicon.txt $dir/lexicon.txt

# An extra question will be added by including the silence phones in one class.
cat $dir/silence_phones.txt| awk '{printf("%s ", $1);} END{printf "\n";}' > $dir/extra_questions.txt || exit 1;

# This is just for diagnostics:
cat data/local/annotations/train.txt  | \
  awk '{for (n=6;n<=NF;n++){ count[$n]++; } } END { for(n in count) { print count[n], n; }}' | \
  sort -nr > $dir/word_counts

awk '{print $1}' $dir/lexicon.txt | \
  perl -e '($word_counts)=@ARGV;
   open(W, "<$word_counts")||die "opening word-counts $word_counts";
   while(<STDIN>) { chop; $seen{$_}=1; }
   while(<W>) {
     ($c,$w) = split;
     if (!defined $seen{$w}) { print; }
   } ' $dir/word_counts > $dir/oov_counts.txt

echo "*Highest-count OOVs are:"
head -n 20 $dir/oov_counts.txt

utils/validate_dict_dir.pl $dir
