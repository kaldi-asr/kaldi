#!/bin/bash

mkdir -p data/lang data/local/dict

cat asr_turkish/lang/lexicon.txt | sed '1,4d' > data/local/dict/lexicon_words.txt

cp asr_turkish/lang/lexicon.txt data/local/dict/.

cp asr_turkish/lang/nonsilence_phones.txt data/local/dict/.

cp asr_turkish/lang/optional_silence.txt data/local/dict/.

touch data/local/dict/extra_questions.txt

echo "SIL" > data/local/dict/silence_phones.txt

echo "<UNK>" > data/lang/oov.txt

echo "Dictionary preparation succeeded"
