#!/usr/bin/env bash
# Copyright (c) 2017, Johns Hopkins University (Jan "Yenda" Trmal<jtrmal@gmail.com>)
# License: Apache 2.0

# Begin configuration section.
tag_percentage=0.1
unk="<unk>"

. ./utils/parse_options.sh

if [ $# -eq 0 ]; then
  echo "Usage: ./local/prepare_lexicon.sh <dataset_dir/text> <olex>"
  exit 1
fi

# End configuration section
set -e -o pipefail
set -o nounset                              # Treat unset variables as an error

text=$1
out=$2

mkdir -p $out
local/prepare_training_text.pl "$unk" $text > ${text}.clean
mv $text ${text}.orig
mv ${text}.clean $text
utils/fix_data_dir.sh `dirname $text`

cut -f 2- -d ' ' $text | perl -ape 's/ /\n/g;' | sort -u > $out/word_list.raw
(echo SIL; grep "<" $out/word_list.raw) | awk '{print $0, $0;}' > $out/silence_lexicon.txt
grep -v "<" $out/word_list.raw > $out/word_list.txt


local/lexicon/make_unicode_lexicon.py --tag-percentage $tag_percentage \
  --silence-lexicon $out/silence_lexicon.txt \
  $out/word_list.txt $out/lexicon.txt $out/grapheme_map.txt

local/prepare_unicode_dict.py --silence-lexicon $out/silence_lexicon.txt \
  $out/lexicon.txt $out

cp $out/lexicon.txt $out/filtered_lexicon.txt

utils/prepare_lang.sh --share-silence-phones true \
  data/local "$unk" data/local/tmp.lang data/lang

