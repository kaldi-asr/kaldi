#!/usr/bin/env bash
# Copyright (c) 2017, Johns Hopkins University (Jan "Yenda" Trmal<jtrmal@gmail.com>)
#               2017  Vimal Manohar
# License: Apache 2.0

# This script prepares the 1997 English Broadcast News (HUB4) corpus.
# /export/corpora/LDC/LDC98S71 
# /export/corpora/LDC/LDC98T28

# Begin configuration section.
# End configuration section
set -e -o pipefail
set -o nounset             # Treat unset variables as an error

if [ $# -ne 3 ]; then
  echo "Usage: $0 <text-source> <speech-source> <out-dir>"
  echo " e.g.: $0 /export/corpora/LDC/LDC98T28/hub4e97_trans_980217 /export/corpora/LDC/LDC98S71/97_eng_bns_hub4 data/local/data/train_bn97"
  exit 1
fi

text_source_dir=$1    # /export/corpora/LDC/LDC98T28/hub4e97_trans_980217
speech_source_dir=$2  # /export/corpora/LDC/LDC98S71/97_eng_bns_hub4
out=$3

mkdir -p $out;

ls $text_source_dir/transcrp/*.sgml > $out/text.list
ls $speech_source_dir/*.sph > $out/audio.list

if [ ! -s $out/text.list ] || [ ! -s $out/audio.list ]; then
  echo "$0: Could not get text and audio files"
  exit 1
fi

local/data_prep/parse_sgm_1997_hub4_eng.pl $out/text.list > \
  $out/transcripts.txt 2> $out/parse_sgml.log || exit 1

if [ ! -s $out/transcripts.txt ]; then
  echo "$0: Could not parse SGML files in $out/text.list"
  exit 1
fi

echo "$0: 1997 English Broadcast News training data (HUB4) prepared in $out"
exit 0
