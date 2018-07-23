#! /bin/bash
# Copyright 2016  Johns Hopkins University (Author: Matthew Wiesner)
# Apache 2.0

encoding="utf_8"
max_len=3

. ./utils/parse_options.sh

command -v morfessor-segment >/dev/null 2>&1 || { echo >&2 "Morfessor seems to either not be installed or not on path."; exit 1; }

if [ $# -ne 3 ]; then
  echo >&2 "Usage: ./local/apply_morphs.sh [opts] <morphs_model_dir> <wordlist> <_word2baseform>"
  echo >&2 "  --encoding <encoding_type> # lexicon encoding"
  echo >&2 "  --max-len <int> # maximum allowed morpheme length"  
  exit 1
fi

model=$1 # data/local/morphs
word_list=$2 # data/local/wordlist.txt
word2baseform=$3 # data/local/morphs/word2baseform

[ ! -d $model ] && echo >&2 "Model directory does not exist." && exit 1
[ ! -f $word_list ] && echo >&2 "Word list does not exist." && exit 1

morfessor-segment --encoding=$encoding --output-format-separator '.' --viterbi-maxlen $max_len \
      -l $model/model.bin $word_list \
      | sed 's/\.[\_\-]\././g' > $model/morphs.txt

paste $word_list $model/morphs.txt > $word2baseform

echo "Created word2baseform file using morphemic decomposition of words as baseforms."
