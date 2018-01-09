#!/bin/bash

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

# awk command from http://stackoverflow.com/questions/2626274/print-all-but-the-first-three-columns
echo 'Gathering missing words...'
cat data/*/train/text | \
  local/count_oovs.pl $lexicon | \
  awk '{for(i=4; i<NF; i++) printf "%s",$i OFS; if(NF) printf "%s",$NF; printf ORS}' | \
  perl -ape 's/\s/\n/g;' | \
  sort | uniq > $workdir/missing.txt
cat $workdir/missing.txt | \
  grep "^[a-z]*$"  > $workdir/missing_onlywords.txt

echo 'Synthesizing pronunciations for missing words...'
phonetisaurus-apply --nbest $var_counts --model $model --thresh 5 --accumulate --word_list $workdir/missing_onlywords.txt > $workdir/missing_g2p_${var_counts}.txt 

echo "Adding new pronunciations to $lexicon"
cat "$lexicon" $workdir/missing_g2p_${var_counts}.txt | sort | uniq > $outlexicon
