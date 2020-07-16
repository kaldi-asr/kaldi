#!/usr/bin/env bash

# Copyright 2016  Allen Guo
#           2017  Xiaohui Zhang
# Apache License 2.0

# This script applies a trained Phonetisarus G2P model to
# synthesize pronunciations for missing words (i.e., words in
# transcripts but not the lexicon), and output the expanded lexicon.

var_counts=1

. ./path.sh || exit 1
. parse_options.sh || exit 1;

if [ $# -ne "4" ]; then
  echo "Usage: $0 <g2p-model> <g2p-tmp-dir> <current-lexicon> <output-lexicon>"
  exit 1
fi

model=$1
workdir=$2
lexicon=$3
outlexicon=$4

mkdir -p $workdir

echo 'Synthesizing pronunciations for missing words...'
phonetisaurus-apply --nbest $var_counts --model $model --thresh 5 --accumulate --word_list $workdir/missing_onlywords.txt > $workdir/missing_g2p_${var_counts}.txt 

echo "Adding new pronunciations to $lexicon"
cat "$lexicon" $workdir/missing_g2p_${var_counts}.txt | sort | uniq > $outlexicon
