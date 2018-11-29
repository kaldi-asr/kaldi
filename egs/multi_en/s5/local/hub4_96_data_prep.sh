#!/bin/bash

###########################################################################################
# This script was copied from egs/hub4_english/s5/local/data_prep/prepare_1996_bn_data.sh
# The source commit was 191ae0a6e5db19d316c82a78c746bcd56cc2d7da
# Changes in lower level script/dir names were made
###########################################################################################

#!/bin/bash
# Copyright (c) 2017, Johns Hopkins University (Jan "Yenda" Trmal<jtrmal@gmail.com>)
#               2017  Vimal Manohar
# License: Apache 2.0

# This script prepares the 1996 English Broadcast News (HUB4) corpus.
# /export/corpora/LDC/LDC97S44 
# /export/corpora/LDC/LDC97T22

# Begin configuration section.
# End configuration section
set -e -o pipefail
set -o nounset             # Treat unset variables as an error

if [ $# -ne 3 ]; then
  echo "Usage: $0 <text-source> <speech-source> <out-dir>"
  echo " e.g.: $0 /export/corpora/LDC/LDC97T22/hub4_eng_train_trans /export/corpora/LDC/LDC97S44/data data/local/data/train_bn96"
  exit 1
fi

text_source_dir=$1    # /export/corpora/LDC/LDC97T22/hub4_eng_train_trans
speech_source_dir=$2  # /export/corpora/LDC/LDC97S44/data
out=$3

mkdir -p $out;

ls $text_source_dir/*/*.txt > $out/text.list
ls $speech_source_dir/*.sph > $out/audio.list

if [ ! -s $out/text.list ] || [ ! -s $out/audio.list ]; then
  echo "$0: Could not get text and audio files"
  exit 1
fi

local/hub4_96_parse_sgm.pl $out/text.list > \
  $out/transcripts.txt 2> $out/parse_sgml.log || exit 1

if [ ! -s $out/transcripts.txt ]; then
  echo "$0: Could not parse SGML files in $out/text.list"
  exit 1
fi

echo "$0: 1996 English Broadcast News training data (HUB4) prepared in $out"
exit 0
