#!/bin/bash

locdata=$1; shift
locdict=$1; shift


mkdir -p $locdict 

perl local/phonetic_transcription_cs.pl $locdata/vocab-full.txt $locdict/cs_transcription.txt

echo "--- Searching for OOV words ..."
gawk 'NR==FNR{words[$1]; next;} !($1 in words)' \
  $locdict/cs_transcription.txt $locdata/vocab-full.txt |\
  egrep -v '<.?s>' > $locdict/vocab-oov.txt

gawk 'NR==FNR{words[$1]; next;} ($1 in words)' \
  $locdata/vocab-full.txt $locdict/cs_transcription.txt |\
  egrep -v '<.?s>' > $locdict/lexicon.txt

wc -l $locdict/vocab-oov.txt
wc -l $locdict/lexicon.txt
