#!/bin/bash
#
# Copyright  2014 Nickolay V. Shmyrev
#            2014 Brno University of Technology (Author: Karel Vesely)
#            2016 Daniel Galvez
#            2016 Vincent Nguyen
# Apache 2.0
#

dir=data/local/dict_nosp
mkdir -p $dir

srcdict=db/TEDLIUM_release-3/TEDLIUM.152k.dic

[ ! -r $srcdict ] && echo "Missing $srcdict" && exit 1

# Join dicts and fix some troubles
cat $srcdict | grep -v -w "<s>" | grep -v -w "</s>" | grep -v -w "<unk>" | \
  LANG= LC_ALL= sort | sed 's:([0-9])::g' > $dir/lexicon_words.txt

cat $dir/lexicon_words.txt | awk '{ for(n=2;n<=NF;n++){ phones[$n] = 1; }} END{for (p in phones) print p;}' | \
  grep -v SIL | sort > $dir/nonsilence_phones.txt

( echo SIL; echo NSN ) > $dir/silence_phones.txt

echo SIL > $dir/optional_silence.txt

# No "extra questions" in the input to this setup, as we don't
# have stress or tone.
echo -n >$dir/extra_questions.txt

# Add to the lexicon the silences, noises etc.
# Typically, you would use "<UNK> NSN" here, but the Cantab Research language models
# use <unk> instead of <UNK> to represent out of vocabulary words.
echo '<unk> NSN' | cat - $dir/lexicon_words.txt | sort | uniq > $dir/lexicon.txt

# Check that the dict dir is okay!
utils/validate_dict_dir.pl $dir || exit 1
