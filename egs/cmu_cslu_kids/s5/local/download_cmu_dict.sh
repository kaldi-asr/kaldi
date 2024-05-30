#!/usr/bin/env bash
# Copyright 2019 Fei Wu
set -eu 
# Adapted from the local/prepare_dict script in 
# the librispeech recipe. Download and prepare CMU_dict.
# For childresn speech ASR tasks, since the vocabulary in cmu_kids and 
# cslu_kids is relatively easy comparing to librispeech, we use only the 
# CMU_dict, and do not handle OOV with G2P.
# Should be run from egs/cmu_cslu_kids.
# Usage:
#   local/download_cmu_dict.sh --dict_dir <path_to_dict_dir>

dict_dir=data/local/dict
OOV="<UNK>"

. ./utils/parse_options.sh || exit 1;
. ./path.sh || exit 1

if [ ! -d $dict_dir ]; then
  echo "Downloading and preparing CMU dict"
  svn co -r 12440 https://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict $dict_dir/raw_dict || exit 1;
  
  echo "Removing the pronunciation variant markers ..."
  grep -v ';;;' $dict_dir/raw_dict/cmudict.0.7a | \
  perl -ane 'if(!m:^;;;:){ s:(\S+)\(\d+\) :$1 :; print; }' | \
  sort -u > $dict_dir/lexicon.txt || exit 1;

  tr -d '\r' <  $dict_dir/raw_dict/cmudict.0.7a.symbols > $dict_dir/nonsilence_phones.txt
  
  echo "$OOV SIL" >> $dict_dir/lexicon.txt
  
  echo "SIL" > $dict_dir/silence_phones.txt
  echo "SPN" >> $dict_dir/silence_phones.txt
  echo "SIL" > $dict_dir/optional_silence.txt

  rm -rf $dict_dir/raw_dict
fi
