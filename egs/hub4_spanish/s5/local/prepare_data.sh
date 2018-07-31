#!/bin/bash
# Copyright (c) 2017, Johns Hopkins University (Jan "Yenda" Trmal<jtrmal@gmail.com>)
# License: Apache 2.0

# Begin configuration section.
# End configuration section
set -e -o pipefail
set -o nounset             # Treat unset variables as an error

out="${@: -1}"             # last argument of the command line
corpus="${@:1:$#-1}"    # first to last-1 arguments

mkdir -p $out;

for src in $corpus; do
  if [ -f $src ]; then
    [[ $src == *.sgm ]] && echo "$src"
  else
    find -L $src -iname "*.sgm"
  fi
done > $out/text.list

for src in $corpus; do
  if [ -f $src ]; then
    [[ $src == *.sph ]] && echo "$src"
  else
    find -L $src -iname "*.sph"
  fi
done  > $out/audio.list

local/parse_sgm.pl $out/text.list > $out/transcript.txt 2> $out/transcript.log

local/write_kaldi_files.pl $out/audio.list $out/transcript.txt $out
utils/utt2spk_to_spk2utt.pl $out/utt2spk > $out/spk2utt
utils/fix_data_dir.sh $out
utils/validate_data_dir.sh --no-feats $out


