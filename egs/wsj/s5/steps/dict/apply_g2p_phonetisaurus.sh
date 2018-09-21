#!/bin/bash
# Copyright 2014  Johns Hopkins University (Author: Yenda Trmal)
# Copyright 2016  Xiaohui Zhang
#           2018  Ruizhe Huang
# Apache 2.0

# This script applies a trained Phonetisarus G2P model to
# synthesize pronunciations for missing words (i.e., words in
# transcripts but not the lexicon), and output the expanded lexicon.

# Begin configuration section.  
stage=0
nbest=1    # Generate up to N variants
pmass=     # Generate so many variants to produce 90 % of the prob mass
model=
# End configuration section.

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. utils/parse_options.sh || exit 1;

set -u
set -e

if [ $# != 3 ]; then
  echo "Usage: $0 [options] <g2p-model> <word-list> <lexicon-out>"
  echo "... where <g2p-model> is the trained g2p model"
  echo "          <word-list> is a list of words whose pronunciation is to be generated"
  echo "          <lexicon-out> output lexicon, whose format is ...."  # TODO
  echo "e.g.: $0 exp/g2p/model.fst exp/g2p/oov_words.txt exp/g2p/model.fst data/local/dict_nosp/lexicon.txt"
  echo ""
  echo "main options (for others, see top of script file)"
  echo "  --nbest <int>    # Maximum number of hypotheses to produce. By default, nbest=1."
  echo "  --pmass <float>  # Select the maximum number of hypotheses summing to pmass total mass"
  echo "                   # for a word. By default, pmass is disabled."
  exit 1;
fi

model=$1
word_list=$2
out_lexicon=$3


if [ -z $pmass ]; then
  echo "Synthesizing pronunciations for words in $word_list based on nbest = $nbest"
  option="--nbest $nbest"
else
  echo "Synthesizing pronunciations for words in $word_list based on pmass = $pmass"
  option="--pmass $pmass"
fi
phonetisaurus-apply $option --model $model --thresh 5 --accumulate --word_list $word_list > $out_lexicon

echo "Finished. Synthesized lexicon for new words is in $out_lexicon"

exit 0
