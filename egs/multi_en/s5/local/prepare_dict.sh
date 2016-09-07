#!/bin/bash

# Copyright 2016  Allen Guo
# Apache License 2.0

# This script prepares data/local/dict_nosp.
# The top part of this file (up to lexicon1.txt) is based on egs/fisher_swbd/s5/local/fisher_swbd_prepare_dict.sh.

# Explanation of lexicon stages:
# - lexicon1.txt: original cmudict with no stress markings or comments
# - lexicon2.txt: punctuation words removed (use this to train G2P)
# - lexicon3.txt: silence phones added
# - lexicon4.txt: unk added
# - lexicon.txt:  either lexicon4.txt with tedlium dict merged in or same as lexicon4.txt

# To be run from one directory above this script.

. ./path.sh

[ $# -gt 1 ] && echo "Usage: $0 [<tedlium-src-dir>]" && exit 1;

dir=data/local/dict_nosp
mkdir -p $dir
echo "Getting CMU dictionary"
svn co  https://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict  $dir/cmudict

# silence phones, one per line. 
for w in sil laughter noise oov breath smack cough; do echo $w; done > $dir/silence_phones.txt
echo sil > $dir/optional_silence.txt

# For this setup we're discarding stress.
cat $dir/cmudict/cmudict.0.7a.symbols | sed s/[0-9]//g | \
 tr '[A-Z]' '[a-z]' | perl -ane 's:\r::; print;' | sort -u > $dir/nonsilence_phones.txt

# An extra question will be added by including the silence phones in one class.
cat $dir/silence_phones.txt| awk '{printf("%s ", $1);} END{printf "\n";}' > $dir/extra_questions.txt || exit 1;

grep -v ';;;' $dir/cmudict/cmudict.0.7a |  tr '[A-Z]' '[a-z]' | \
 perl -ane 'if(!m:^;;;:){ s:(\S+)\(\d+\) :$1 :; s:  : :; print; }' | \
   sed s/[0-9]//g | sort -u > $dir/lexicon1.txt || exit 1;

# Get rid of punctuation words
patch -o $dir/lexicon2.txt $dir/lexicon1.txt local/dict.patch

# Add prons for silence phones
for w in `grep -v sil $dir/silence_phones.txt`; do
  echo "[$w] $w"
done | cat - $dir/lexicon2.txt > $dir/lexicon3.txt || exit 1;

# Add <unk>
echo "<unk> oov" | cat - $dir/lexicon3.txt > $dir/lexicon4.txt

# Add in tedlium dict, if available
if [ $# == 1 ]; then
  cat $1/TEDLIUM.152k.dic | egrep '\S+\s+.+' | \
    grep -v -w "<s>" | grep -v -w "</s>" | grep -v -w "<unk>" | grep -v 'ERROR' | \
    sed 's:([0-9])::g' |sed 's:\s\+: :g' | tr A-Z a-z | \
    cat - $dir/lexicon4.txt | sort -u > $dir/lexicon.txt || exit 1;
else
  cat $dir/lexicon4.txt | sort -u > $dir/lexicon.txt
fi
