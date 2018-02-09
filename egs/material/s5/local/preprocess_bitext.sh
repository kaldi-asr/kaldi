#!/bin/bash

set -euo pipefail
set -e -o pipefail                                                              
set -o nounset                              # Treat unset variables as an error 
echo "$0 $@"

language=swahili
srctext_bitext=data/bitext/text

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

set -e -o pipefail
set -o nounset                              # Treat unset variables as an error

language_affix=sw
if [ "$language" == "tagalog" ]; then language_affix="tl"; fi
MOSES=/home/pkoehn/moses
SOURCE_TC_MODEL=/home/pkoehn/experiment/material-${language_affix}-en/truecaser/truecase-model.1.${language_affix}

# Normalize punctuation and tokenize input
$MOSES/scripts/tokenizer/normalize-punctuation.perl ${language_affix} < ${srctext_bitext} \
 | $MOSES/scripts/tokenizer/tokenizer.perl -a -l ${language_affix} > ${srctext_bitext}.tok

# Truecase
$MOSES/scripts/recaser/truecase.perl -model $SOURCE_TC_MODEL \
  < ${srctext_bitext}.tok > ${srctext_bitext}.tc

# Remove punctuation
cat ${srctext_bitext}.tc | sed 's/&apos; //g' | sed 's/&apos//g' | sed 's/&#91//g' | sed 's/&#93//g' | sed 's/&quot; //g' | sed 's/&quot //g' | sed 's/&amp; //g' | sed 's/@-@ //g' | sed 's/-//g' | sed 's/://g' | sed 's/\///g' | sed 's/%//g' | sed 's/+//g' | sed 's/( //g' | sed 's/) //g' | sed 's/\, //g' | sed 's/ \.//g' | sed 's/\?//g' | sed 's/\!//g' | sed 's/\;//g'

