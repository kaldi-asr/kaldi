#!/usr/bin/env bash

# Copyright 2016  Allen Guo
#           2017  Xiaohui Zhang
#           2017  Intellisist, Inc. (Author: Navneeth K)
# Apache License 2.0

# This script prepares data/local/dict_cmu_tedlium (combination of CMUDict and the TEDLIUM (Cantab) lexicon).
# The top part of this file (up to lexicon1.txt) is based on egs/fisher_swbd/s5/local/fisher_swbd_prepare_dict.sh.
# But we use cmudict-0.7b rather than cmudict.0.7a

# To be run from one directory above this script.

. ./path.sh

# The parts of the output of this that will be needed are
# [in data/local/dict/ ]
# lexicon.txt
# extra_questions.txt
# nonsilence_phones.txt
# optional_silence.txt
# silence_phones.txt

#check existing directories
[ $# -gt 1 ] && echo "Usage: $0 [<tedlium-src-dir>]" && exit 1;

isuconv=`which uconv` 
if [ -z $isuconv ]; then
  echo "uconv was not found. You must install the icu4c package."
  exit 1;
fi


dir=data/local/dict_cmu_tedlium
mkdir -p $dir
echo "Getting CMU dictionary"
svn co  https://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict  $dir/cmudict

# silence phones, one per line. 
for w in sil laughter noise oov; do echo $w; done > $dir/silence_phones.txt
echo sil > $dir/optional_silence.txt

# For this setup we're discarding stress.
cat $dir/cmudict/cmudict-0.7b.symbols | sed s/[0-9]//g | \
 tr '[A-Z]' '[a-z]' | perl -ane 's:\r::; print;' | sort | uniq > $dir/nonsilence_phones.txt

# An extra question will be added by including the silence phones in one class.
cat $dir/silence_phones.txt| awk '{printf("%s ", $1);} END{printf "\n";}' > $dir/extra_questions.txt || exit 1;

grep -v ';;;' $dir/cmudict/cmudict-0.7b | uconv -f iso-8859-1  -t utf-8 -x Any-NFC - |  tr '[A-Z]' '[a-z]' | \
 perl -ane 'if(!m:^;;;:){ s:(\S+)\(\d+\) :$1 :; s:  : :; print; }' | \
 perl -ane '@A = split(" ", $_); for ($n = 1; $n<@A;$n++) { $A[$n] =~ s/[0-9]//g; } print join(" ", @A) . "\n";' | \
 sort | uniq > $dir/lexicon1.txt || exit 1;

# Get rid of punctuation words
patch -o $dir/lexicon2.txt $dir/lexicon1.txt local/dict.patch

# Add prons for silence phones, and noise
(echo "[sil] sil"; echo "[laughter] laughter"; echo "[noise] noise"; \
  echo "<unk> oov"; echo "[vocalized-noise] oov";) | cat - $dir/lexicon2.txt > $dir/lexicon3.txt

# Add in tedlium dict, if available
if [ $# == 1 ]; then
  cat $1/TEDLIUM.152k.dic | uconv -f utf-8  -t utf-8 -x Any-NFC - | egrep '\S+\s+.+' | \
    grep -v -w "<s>" | grep -v -w "</s>" | grep -v -w "<unk>" | grep -v 'ERROR' | \
    sed 's:([0-9])::g' |sed 's:\s\+: :g' | tr A-Z a-z | \
    cat - $dir/lexicon3.txt | sort -u > $dir/lexicon.txt || exit 1;
else
  cat $dir/lexicon3.txt | sort -u > $dir/lexicon.txt
fi

utils/validate_dict_dir.pl $dir 
