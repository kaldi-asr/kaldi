#!/bin/bash
# Copyright (c) 2017, Johns Hopkins University (Jan "Yenda" Trmal<jtrmal@gmail.com>)
# License: Apache 2.0

# Begin configuration section.
tag_percentage=0.1

# End configuration section
set -e -o pipefail
set -o nounset                              # Treat unset variables as an error

text=data/train/text
out=data/local
unk="<unk>"

mkdir -p $out
local/prepare_training_text.pl "$unk" $text > ${text}.clean
mv $text ${text}.orig
mv ${text}.clean $text
utils/fix_data_dir.sh `dirname $text`

cut -f 2- -d ' ' $text | sed 's/ /\n/g' | sort -u > $out/word_list.raw
(echo SIL; grep "<" $out/word_list.raw) | awk '{print $0, $0;}' > $out/nonspeech.txt
grep -v "<" $out/word_list.raw > $out/word_list.txt


local/lexicon/make_unicode_lexicon.py --tag_percentage $tag_percentage \
  --fmt "word_list" --nonspeech $out/nonspeech.txt \
  --extraspeech $out/extraspeech.txt --verbose \
  $out/word_list.txt $out/lexicon.txt

local/prepare_unicode_dict.py --nonspeech $out/nonspeech.txt \
  --extraspeech $out/extraspeech.txt $out/lexicon_table.txt $out/

cp $out/lexicon.txt $out/filtered_lexicon.txt

utils/prepare_lang.sh --share-silence-phones true \
  data/local "$unk" data/local/tmp.lang data/lang

