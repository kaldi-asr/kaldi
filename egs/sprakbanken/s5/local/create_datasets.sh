#!/bin/bash

# Copyright 2014 Mirsk Digital ApS  (Author: Andreas Kirkedal)

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.

if [ $# != 2 ]; then
  echo "Usage: create_dataset.sh <src-data-dir> <dest-dir> "
  exit 1
fi


src=$1
dest=$2
mkdir $dest
python local/normalize_transcript_prefixed.py local/norm_dk/numbersLow.tbl $src/text.unnormalised $src/onlyids $src/transcripts.am 
local/norm_dk/format_text.sh am $src/transcripts.am > $src/onlytext
paste -d ' ' $src/onlyids $src/onlytext > $dest/text
for f in wav.scp utt2spk; do
    cp $src/$f $dest/$f
done
utils/utt2spk_to_spk2utt.pl $dest/utt2spk > $dest/spk2utt
utils/validate_data_dir.sh --no-feats $dest || exit 1;
