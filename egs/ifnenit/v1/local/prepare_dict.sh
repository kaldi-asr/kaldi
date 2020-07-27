#!/usr/bin/env bash
. ./cmd.sh
. ./path.sh

# This script is originally from qatip project (http://qatsdemo.cloudapp.net/qatip/demo/)
# of Qatar Computing Research Institute (http://qcri.qa/)

# To be run from one directory above this script.
# Prepare the dict folder. 
# Creating lexicon.txt, phonemeset, nonsilence_phones.txt, extra_questions.txt and silence_phones.txt.

if [ -d "data/local" ]; then
  rm -r data/local
fi

## Determine phoneme set
mkdir -p data/local/lm
cat data/train/text | cut -d' ' -f2- | tr ' ' "\n" | sort -u > data/local/lm/train.vocab
cat data/local/lm/train.vocab | python3 local/make_latin_words.py > data/train/words2latin
cat data/train/text | cut -d' ' -f2- | python3 local/transcript_to_latin.py data/train/words2latin | cut -d' ' -f2- | tr ' ' "\n" | sort | uniq -c | awk '{if ($1 > 50 || length($2) == 3) print $2}' | fgrep -v '~A' > data/local/phonemeset

## Lexicon and word/phoneme lists
mkdir -p data/lang/
mkdir -p data/local/dict
echo '<unk>' > data/lang/oov.txt
cat data/train/words2latin | python3 local/map_to_rareA.py data/local/phonemeset > data/local/dict/lexicon.txt
echo "<unk> rareA" >> data/local/dict/lexicon.txt
echo "!SIL sil" >> data/local/dict/lexicon.txt

cat data/local/phonemeset | fgrep -v '.A' | fgrep -v ',A' | fgrep -v 'conn' | fgrep -v 'sil' | sort > data/local/dict/nonsilence_phones.txt

echo ',A' > data/local/dict/silence_phones.txt
echo '.A' >> data/local/dict/silence_phones.txt
echo 'conn' >> data/local/dict/silence_phones.txt
echo 'rareA' >> data/local/dict/silence_phones.txt
echo 'sil' >> data/local/dict/silence_phones.txt
echo 'sil' > data/local/dict/optional_silence.txt
# config folder
cat config/extra_questions.txt| python3 local/reduce_to_vocabulary.py data/local/dict/nonsilence_phones.txt | sort -u | fgrep ' ' > data/local/dict/extra_questions.txt

mv data/local/dict/lexicon.txt data/local/dict/prelexicon.txt
# # add ligatures
cat data/local/dict/prelexicon.txt |  sed 's/\s\+la[BM]\{1\}\s\+conn\s\+a[meha]\{1\}E/ laLE/g' | python3 local/add_ligature_variants.py config/ligatures > data/local/dict/lexicon.txt
cat data/local/dict/lexicon.txt| cut -d' ' -f2- | tr ' ' "\n" | sort -u > data/local/phonemeset
cat data/local/phonemeset | fgrep -v 'rare' | fgrep -v '.A' | fgrep -v ',A' | fgrep -v 'conn' | fgrep -v 'sil' | sort > data/local/dict/nonsilence_phones.txt

