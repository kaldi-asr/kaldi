#!/usr/bin/env bash

set -euo pipefail
set -e -o pipefail                                                              
set -o nounset                              # Treat unset variables as an error 
echo "$0 $@"

language=swahili
srctext_bitext=data/bitext/text

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

output=$1

set -e -o pipefail
set -o nounset                              # Treat unset variables as an error

if [ "$language" == "swahili" ]; then
  language_affix="sw"
elif [ "$language" == "tagalog" ]; then
  language_affix="tl"
elif [ "$language" == "somali" ]; then
  language_affix="so"
fi
MOSES=/home/pkoehn/moses

# Normalize punctuation and tokenize input
$MOSES/scripts/tokenizer/normalize-punctuation.perl ${language_affix} < ${srctext_bitext} \
 | $MOSES/scripts/tokenizer/tokenizer.perl -a -l ${language_affix} > ${srctext_bitext}.tok

# convert to lower cases
cat ${srctext_bitext}.tok | tr 'A-Z' 'a-z' > ${srctext_bitext}.tc

# Remove punctuation
cat ${srctext_bitext}.tc | sed 's/&apos; //g' | sed 's/&apos//g' | sed 's/&#91//g' | sed 's/&#93//g' | sed 's/&quot; //g' | sed 's/&quot //g' | sed 's/&amp; //g' | sed 's/@-@ //g' | sed 's/-//g' | sed 's/://g' | sed 's/\///g' | sed 's/%//g' | sed 's/+//g' | sed 's/( //g' | sed 's/) //g' | sed 's/\, //g' | sed 's/ \.//g' | sed 's/\?//g' | sed 's/\!//g' | sed 's/\;//g' > $output

