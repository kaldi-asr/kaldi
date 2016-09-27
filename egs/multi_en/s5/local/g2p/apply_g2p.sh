#!/bin/bash

# Copyright 2016  Allen Guo
# Apache License 2.0

# This script applies a trained Sequitur G2P model to
# synthesize pronunciations for missing words (i.e., words in
# transcripts but not the lexicon). Only missing words that are
# purely alphabetical (no punctuation) are synthesized.

. ./path.sh || exit 1

if [ $# -ne "3" ]; then
  echo "Usage: $0 <g2p-model> <g2p-tmp-dir> <current-lexicon>"
  exit 1
fi

model=$1
workdir=$2
lexicon=$3

mkdir -p $workdir

# awk command from http://stackoverflow.com/questions/2626274/print-all-but-the-first-three-columns
echo 'Gathering missing words...'
cat data/*/train/text | \
  local/count_oovs.pl $lexicon | \
  awk '{for(i=4; i<NF; i++) printf "%s",$i OFS; if(NF) printf "%s",$NF; printf ORS}' | \
  sed 's/\s/\n/g' | \
  grep -v 0 | sort | uniq | \
  grep '^[a-z]*$' > $workdir/missing.txt

echo 'Synthesizing pronunciations for missing words...'
PYTHONPATH=$sequitur_path:$PYTHONPATH $PYTHON $sequitur \
  --model $model --apply $workdir/missing.txt > $workdir/missing_g2p.txt

echo "Adding new pronunciations to $lexicon"
echo "Original lexicon moved to ${lexicon}.bkp"
mv $lexicon "$lexicon".bkp
cat "$lexicon".bkp $workdir/missing_g2p.txt | sort | uniq > $lexicon
