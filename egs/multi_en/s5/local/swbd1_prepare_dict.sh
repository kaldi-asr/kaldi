#!/bin/bash

###########################################################################################
# This script was copied from egs/swbd/s5c/local/swbd1_prepare_dict.sh
# The source commit was e435334b4b7bb9eaf28764f37bb44a825da320fc
# Changes made:
#  - Modified path names of srcdir, dir, MSU_single_letter.txt and dict.patch to match multi_en naming convention.
###########################################################################################

# Formatting the Mississippi State dictionary for use in Edinburgh. Differs
# from the one in Kaldi s5 recipe in that it uses lower-case --Arnab (Jan 2013)

# To be run from one directory above this script.

. ./path.sh

#check existing directories
[ $# != 0 ] && echo "Usage: local/swbd1_data_prep.sh" && exit 1;

srcdir=data/local/swbd  # This is where we downloaded some stuff..
dir=data/local/dict_swbd
mkdir -p $dir
srcdict=$srcdir/swb_ms98_transcriptions/sw-ms98-dict.text

# assume swbd_p1_data_prep.sh was done already.
[ ! -f "$srcdict" ] && echo "$0: No such file $srcdict" && exit 1;

rm -f $dir/lexicon0.txt
cp $srcdict $dir/lexicon0.txt || exit 1;
chmod +w $dir/lexicon0.txt
patch <local/swbd_dict.patch $dir/lexicon0.txt || exit 1;

#(2a) Dictionary preparation:
# Pre-processing (remove comments)
grep -v '^#' $dir/lexicon0.txt | awk 'NF>0' | sort > $dir/lexicon1.txt || exit 1;

cat $dir/lexicon1.txt | awk '{ for(n=2;n<=NF;n++){ phones[$n] = 1; }} END{for (p in phones) print p;}' | \
  grep -v sil > $dir/nonsilence_phones.txt  || exit 1;

( echo sil; echo spn; echo nsn; echo lau ) > $dir/silence_phones.txt

echo sil > $dir/optional_silence.txt

# No "extra questions" in the input to this setup, as we don't
# have stress or tone.
echo -n >$dir/extra_questions.txt

cp conf/MSU_single_letter.txt $dir/
# Add to the lexicon the silences, noises etc.
# Add single letter lexicon
# The original swbd lexicon does not have precise single letter lexicion
# e.g. it does not have entry of W
( echo '!sil sil'; echo '[vocalized-noise] spn'; echo '[noise] nsn'; \
  echo '[laughter] lau'; echo '<unk> spn' ) \
  | cat - $dir/lexicon1.txt $dir/MSU_single_letter.txt  > $dir/lexicon2.txt || exit 1;

# Map the words in the lexicon.  That is-- for each word in the lexicon, we map it
# to a new written form.  The transformations we do are:
# remove laughter markings, e.g.
# [LAUGHTER-STORY] -> STORY
# Remove partial-words, e.g.
# -[40]1K W AH N K EY
# becomes -1K
# and
# -[AN]Y IY
# becomes
# -Y
# -[A]B[OUT]- B
# becomes
# -B-
# Also, curly braces, which appear to be used for "nonstandard"
# words or non-words, are removed, e.g.
# {WOLMANIZED} W OW L M AX N AY Z D
# -> WOLMANIZED
# Also, mispronounced words, e.g.
#  [YEAM/YEAH] Y AE M
# are changed to just e.g. YEAM, i.e. the orthography
# of the mispronounced version.
# Note-- this is only really to be used in training.  The main practical
# reason is to avoid having tons of disambiguation symbols, which
# we otherwise would get because there are many partial words with
# the same phone sequences (most problematic: S).
# Also, map
# THEM_1 EH M -> THEM
# so that multiple pronunciations just have alternate entries
# in the lexicon.

local/swbd1_map_words.pl -f 1 $dir/lexicon2.txt | sort -u \
  > $dir/lexicon3.txt || exit 1;

python local/swbd_format_acronyms_dict.py -i $dir/lexicon3.txt -o $dir/lexicon4.txt \
  -L $dir/MSU_single_letter.txt -M $dir/acronyms_raw.map
cat $dir/acronyms_raw.map | sort -u > $dir/acronyms.map

( echo 'i ay' )| cat - $dir/lexicon4.txt | tr '[A-Z]' '[a-z]' | sort -u > $dir/lexicon5.txt

pushd $dir >&/dev/null
ln -sf lexicon5.txt lexicon.txt # This is the final lexicon.
popd >&/dev/null
rm $dir/lexiconp.txt 2>/dev/null
echo Prepared input dictionary and phone-sets for Switchboard phase 1.
