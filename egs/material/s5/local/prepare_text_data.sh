#!/bin/bash
# Copyright (c) 2017, Johns Hopkins University (Jan "Yenda" Trmal<jtrmal@gmail.com>)
# License: Apache 2.0

# Begin configuration section.
# End configuration section
set -e -o pipefail
set -o nounset                              # Treat unset variables as an error
echo "$0 " "$@"

if [ $# -ne 1 ] ; then
  echo "Invalid number of script parameters. "
  echo "  $0 <path-to-material-corpus>"
  echo "e.g."
  echo "  $0 /export/corpora5/MATERIAL/IARPA_MATERIAL_BASE-1A-BUILD_v1.0/"
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




