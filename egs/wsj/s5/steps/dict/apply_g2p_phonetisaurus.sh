#!/bin/bash
# Copyright 2014  Johns Hopkins University (Author: Yenda Trmal)
# Copyright 2016  Xiaohui Zhang
#           2018  Ruizhe Huang
# Apache 2.0

# This script applies a trained Phonetisarus G2P model to
# synthesize pronunciations for missing words (i.e., words in
# transcripts but not the lexicon), and output the expanded lexicon.
# The user could specify either nbest or pmass option 
# to determine the number of output pronunciation variants, 
# or use them together to get the intersection of two options.

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
  echo "e.g.: $0 --nbest 1 exp/g2p/model.fst exp/g2p/oov_words.txt data/local/dict_nosp/lexicon.txt"
  echo ""
  echo "main options (for others, see top of script file)"
  echo "  --nbest <int>    # Maximum number of hypotheses to produce. By default, nbest=20"
  echo "  --pmass <float>  # Select the maximum number of hypotheses summing to a total mass of pmass amount, within [0, 1], for a word. By default, pmass=1.0"
  exit 1;
fi

model=$1
word_list=$2
out_lexicon=$3
out_lexicon_failed="${out_lexicon}.failed"

[ -z $pmass ] && [ -z $nbest ] && echo "$0: nbest or/and pmass should be specified." && exit 1;

# three options: 1) nbest, 2) pmass, 3) nbest+pmass,
nbest=${nbest:-20}   # if nbest is not specified, set it to 20, due to Phonetisaurus mechanism
pmass=${pmass:-1.0}  # if pmass is not specified, set it to 1.0, due to Phonetisaurus mechanism

[[ ! $nbest =~ ^[1-9][0-9]*$ ]] && echo "$0: nbest should be a positive integer." && exit 1;

echo "$0: Synthesizing pronunciations for words in $word_list based on nbest=$nbest and pmass=$pmass"
phonetisaurus-apply --pmass $pmass --nbest $nbest --model $model --thresh 5 --accumulate --verbose --prob --word_list $word_list \
  1>$out_lexicon 

echo "$0: Completed. Synthesized lexicon for new words is in $out_lexicon"

exit 0
