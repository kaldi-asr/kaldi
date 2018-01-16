#!/bin/bash
# Copyright (c) 2017, Johns Hopkins University (Jan "Yenda" Trmal<jtrmal@gmail.com>)
# License: Apache 2.0

# Begin configuration section.
# End configuration section
set -e -o pipefail
set -o nounset                              # Treat unset variables as an error
echo "$0 " "$@"

if [ $# -ne 2 ] ; then
  echo "Invalid number of script parameters. "
  echo "  $0 <path-to-material-corpus> <language-name>"
  echo "e.g."
  echo "  $0 /export/corpora5/MATERIAL/IARPA_MATERIAL_BASE-1A-BUILD_v1.0/ swahili"
  exit
fi
data=$1;
language=$2
conversational_train=$data/conversational/training/
mkdir -p data/$language/train/
for file in $conversational_train/transcription/*txt ; do
  ./local/parse_transcripts.pl $file
done  > data/$language/train/transcripts.txt


conversational_dev=$data/conversational/dev/
mkdir -p data/$language/dev
for file in $conversational_dev/transcription/*txt ; do
  ./local/parse_transcripts.pl $file
done > data/$language/dev/transcripts.txt


cat data/$language/train/transcripts.txt | \
  local/cleanup_transcripts.pl | \
  local/create_datafiles.pl data/$language/train/

cat data/$language/dev/transcripts.txt | \
  local/cleanup_transcripts.pl | \
  local/create_datafiles.pl data/$language/dev/




