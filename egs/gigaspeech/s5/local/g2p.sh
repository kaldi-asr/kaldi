#!/usr/bin/env bash
# Copyright 2021  Xiaomi Corporation (Author: Yongqing Wang)
#                 Seasalt AI, Inc (Author: Guoguo Chen)

# Generates pronunciations for out-of-vocabulary words using the Sequitur G2P
# toolkit.

set -e
set -o pipefail

. ./path.sh || exit 1;
. ./utils/parse_options.sh || exit 1;

if [ $# -ne 3 ]; then
  echo "Usage: $0 <g2p-model> <words> <lexicon>"
  echo " e.g.: $0 g2p/g2p.model.4 words.list data/local/dict/g2p/lexicon.txt"
  echo ""
  echo "  <g2p-model>   G2P model file."
  echo "  <words>       List of words to generate pronunciations for."
  echo "  <lexicon>     Generated lexicon."
  exit 1;
fi

g2p_model=$1
words=$2
lexicon=$3

[ ! -f $g2p_model ] && echo "$0: Can't G2P model: $g2p_model" && exit 1;
[ ! -f $words ] && echo "$0: Can't find the G2P input: $words" && exit 1;

sequitur=`which g2p.py`
if [ -z $sequitur ] || [ ! -x $sequitur ]; then
  echo "$0: Can't find the Sequitur G2P script. Please check $KALDI_ROOT/tools"
  echo "$0: for installation script and instructions."
  exit 1;
fi

# Applies Sequitur G2P
echo "$0: Applying Sequitur G2P to $words"
PYTHONPATH=$sequitur_path:$PYTHONPATH \
           $sequitur --model=$g2p_model --apply $words > ${lexicon}.tmp || exit 1;

# Turns out Sequitur has some sort of bug and it doesn't output pronunciations
# for some (admittedly peculiar) words. We manually specify these exceptions
# below. More such entries can be added, separated by "\n"
g2p_exceptions="HH HH"
awk 'NR==FNR{p[$1]=$0; next;} {if ($1 in p) print p[$1]; else print}' \
  <(echo -e $g2p_exceptions) ${lexicon}.tmp > $lexicon || exit 1;

rm ${lexicon}.tmp

echo "$0: Done"
