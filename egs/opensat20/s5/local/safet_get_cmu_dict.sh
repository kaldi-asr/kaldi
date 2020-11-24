#!/bin/bash
# Copyright (c) 2020, Johns Hopkins University (Jan "Yenda" Trmal<jtrmal@gmail.com>)
# License: Apache 2.0

# Begin configuration section.
lm_order=6
# End configuration section
. ./utils/parse_options.sh
set -e -o pipefail
set -o nounset                              # Treat unset variables as an error
. ./path.sh

OUTPUT=data/local
mkdir -p $OUTPUT

[ -f data/cmudict-0.7b ] || \
  curl http://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict/cmudict-0.7b > $OUTPUT/cmudict-0.7b

# add cmu dict words in lowercase in the lexicon
uconv -f iso-8859-1 -t utf-8 $OUTPUT/cmudict-0.7b| grep -v ';;' | sed 's/([0-9])//g' | \
  perl -ne '($a, $b) = split " ", $_, 2; $b =~ s/[0-9]//g; $a = lc $a; print "$a $b";' > $OUTPUT/lexicon.txt

# add SIL, <UNK>, %uh, {breath}, {lipsmack}, {laugh}, {cough}, <noise> words in the lexicon 
# <UNK> word is mapped to <unk> phone
# {breath}, {lipsmack}, {laugh}, {cough}, <noise> are mapped to <noise>
mkdir -p $OUTPUT/dict_nosp
echo -e "SIL <sil>\n<UNK> <unk>" |  cat - local/safet_hesitations.txt $OUTPUT/lexicon.txt | sort -u > $OUTPUT/dict_nosp/lexicon1.txt


# add some specific words, those are only with 100 missing occurences or more
# add mm hmm mm-hmm  words in the lexicon
( echo "mm M"; \
  echo "hmm HH M"; \
  echo "mm-hmm M HH M" ) | cat - $OUTPUT/dict_nosp/lexicon1.txt \
     | sort -u > $OUTPUT/dict_nosp/lexicon2.txt

# Add prons for laughter, noise, oov as phones in the silence phones
for w in laughter noise oov; do echo $w; done > $OUTPUT/dict_nosp/silence_phones.txt

# add [laughter], [noise], [oov] words in the lexicon
for w in `grep -v sil $OUTPUT/dict_nosp/silence_phones.txt`; do
  echo "[$w] $w"
done | cat - $OUTPUT/dict_nosp/lexicon2.txt > $OUTPUT/dict_nosp/lexicon.txt


# Add <sil>, <unk>, <noise>, <hes> as phones in the silence phones
echo -e "SIL <sil>\n<UNK> <unk>" |  cat - local/safet_hesitations.txt | cut -d ' ' -f 2- | sed 's/ /\n/g' | \
  sort -u | sed '/^ *$/d' >> $OUTPUT/dict_nosp/silence_phones.txt

echo '<UNK>' > $OUTPUT/dict_nosp/oov.txt

echo '<sil>' > $OUTPUT/dict_nosp/optional_silence.txt


cat $OUTPUT/lexicon.txt | cut -d ' ' -f 2- | sed 's/ /\n/g' | \
  sort -u | sed '/^ *$/d' > $OUTPUT/dict_nosp/nonsilence_phones.txt

