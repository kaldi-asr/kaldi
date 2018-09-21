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
nbest=          # Generate up to $nbest variants
pmass=          # Generate so many variants to produce $pmass ammount, like 90%, of the prob mass
# End configuration section.

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. utils/parse_options.sh || exit 1;

set -u
set -e

if [ $# != 3 ]; then
  echo "Usage: $0 [options] <g2p-model> <word-list> <lexicon-out>"
  echo "... where <g2p-model> is the trained g2p model."
  echo "          <word-list> is a list of words whose pronunciation is to be generated."
  echo "          <lexicon-out> output lexicon, whose format is <word>\t<prob>\t<pronunciation> for each line."
  echo "e.g.: $0 exp/g2p/model.fst exp/g2p/oov_words.txt data/local/dict_nosp/lexicon.txt"
  echo ""
  echo "main options (for others, see top of script file)"
  echo "  --nbest <int>    # Maximum number of hypotheses to produce. By default, nbest=1."
  echo "  --pmass <float>  # Select the maximum number of hypotheses summing to a total mass of pmass amount, within [0, 1], for a word."
  echo "  --nbest <int> --pmass <float>  # When specified together, we generate the intersection of these two options."
  exit 1;
fi

model=$1
word_list=$2
out_lexicon=$3

[ ! -z $nbest ] && [[ ! $nbest =~ ^[0-9]+$ ]] && echo "$0: nbest should be a positive integer." && exit 1
[ ! -z $pmass ] && ! { [[ $pmass =~ ^[0-9]+\.?[0-9]*$ ]] && [ $(bc <<< "$pmass >= 0") -eq 1 -a $(bc <<< "$pmass <= 1") -eq 1 ]; } \
  && echo "$0: pmass should be within [0, 1]." && exit 1
[ -z $pmass ] && [ -z $nbest ] && nbest=1

if [ -z $pmass ]; then
  echo "Synthesizing pronunciations for words in $word_list based on nbest=$nbest"
  options="--nbest $nbest --pmass 1.0"
elif [ -z $nbest ]; then
  echo "Synthesizing pronunciations for words in $word_list based on pmass=$pmass"
  options="--pmass $pmass --nbest 20"
else
  echo "Synthesizing pronunciations for words in $word_list based on nbest=$nbest and pmass=$pmass"
  options="--pmass $pmass --nbest $nbest"
fi
phonetisaurus-apply $options --model $model --thresh 5 --accumulate --verbose --prob --word_list $word_list 1>$out_lexicon

echo "Finished. Synthesized lexicon for new words is in $out_lexicon"

exit 0
