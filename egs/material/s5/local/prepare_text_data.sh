#!/usr/bin/env bash
# Copyright (c) 2017, Johns Hopkins University (Jan "Yenda" Trmal<jtrmal@gmail.com>)
# License: Apache 2.0

# Begin configuration section.
# End configuration section
set -e -o pipefail
set -o nounset                              # Treat unset variables as an error
echo "$0 " "$@"

language=swahili

. ./utils/parse_options.sh

if [ $# -ne 1 ] ; then
  echo "Invalid number of script parameters. "
  echo "  $0 [options] <path-to-material-corpus>"
  echo "e.g."
  echo "  $0 --language swahili /export/corpora5/MATERIAL/IARPA_MATERIAL_BASE-1A-BUILD_v1.0/"
  exit
fi
data=$1;
conversational_train=$data/conversational/training/
mkdir -p data/train/
for file in $conversational_train/transcription/*txt ; do
  ./local/parse_transcripts.pl $file
done  > data/train/transcripts.txt


conversational_dev=$data/conversational/dev/
mkdir -p data/dev
for file in $conversational_dev/transcription/*txt ; do
  ./local/parse_transcripts.pl $file
done > data/dev/transcripts.txt


cat data/train/transcripts.txt | \
  local/cleanup_transcripts.pl | \
  local/create_datafiles.pl data/train/

cat data/dev/transcripts.txt | \
  local/cleanup_transcripts.pl | \
  local/create_datafiles.pl data/dev/

if [ "$language" == "swahili" ]; then
  language_affix="sw"
elif [ "$language" == "tagalog" ]; then
  language_affix="tl"
elif [ "$language" == "somali" ]; then
  language_affix="so"
fi
MOSES=/home/pkoehn/moses
SOURCE_TC_MODEL=/home/pkoehn/experiment/material-asr-${language_affix}-en/truecaser/truecase-model.1.${language_affix}

for i in train dev; do
  cat data/$i/text | cut -d " " -f2- > data/$i/text.notruecase
  cat data/$i/text | cut -d " " -f1  > data/$i/uttids
  # Truecase
  $MOSES/scripts/recaser/truecase.perl -model $SOURCE_TC_MODEL \
    < data/$i/text.notruecase | sed "s=<= <=g" > data/$i/text.truecase
#  cat data/$i/text.truecase | sed 's/&apos; //g' | sed 's/&apos//g' | sed 's/&#91//g' | sed 's/&#93//g' | sed 's/&quot; //g' | sed 's/&quot //g' | sed 's/&amp; //g' | sed 's/@-@ //g' | sed 's/://g' | sed 's/\///g' | sed 's/%//g' | sed 's/+//g' | sed 's/( //g' | sed 's/) //g' | sed 's/\, //g' | sed 's/ \.//g' | sed 's/\?//g' | sed 's/\!//g' | sed 's/\;//g' > data/$i/text.nopunc
  cat data/$i/text.truecase | tr 'A-Z' 'a-z' > data/$i/text.nopunc
  paste -d " " data/$i/uttids data/$i/text.nopunc > data/$i/text
done


