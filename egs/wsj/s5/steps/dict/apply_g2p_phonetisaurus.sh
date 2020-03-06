#!/usr/bin/env bash
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
nbest=      # Generate up to N, like N=3, pronunciation variants for each word
            # (The maximum size of the nbest list, not considering pruning and taking the prob-mass yet). 
thresh=5    # Pruning threshold for the n-best list, in (0, 99], which is a -log-probability value.
            # A large threshold makes the nbest list shorter, and less likely to hit the max size.
            # This value corresponds to the weight_threshold in shortest-path.h of openfst.
pmass=      # Select the top variants from the pruned nbest list,
            # summing up to this total prob-mass for a word.
            # On the "boundary", it's greedy by design, e.g. if pmass = 0.8,
            # and we have prob(pron_1) = 0.5, and prob(pron_2) = 0.4, then we get both.
# End configuration section.

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. utils/parse_options.sh || exit 1;

set -u
set -e

if [ $# != 3 ]; then
  echo "Usage: $0 [options] <word-list> <g2p-model-dir> <output-dir>"
  echo "... where <word-list> is a list of words whose pronunciation is to be generated."
  echo "          <g2p-model-dir> is a directory used as a target during training of G2P"
  echo "          <output-dir> is the directory where the output lexicon should be stored."
  echo "                       The format of the output lexicon output-dir/lexicon.lex is" 
  echo "                       <word>\t<prob>\t<pronunciation> per line."
  echo "e.g.: $0 --nbest 1 exp/g2p/oov_words.txt exp/g2p exp/g2p/oov_lex"
  echo ""
  echo "main options (for others, see top of script file)"
  echo "  --nbest <int>    # Generate upto N pronunciation variants for each word." 
  echo "  --pmass <float>  # Select the top variants from the pruned nbest list," 
  echo "                   # summing up to this total prob-mass, within [0, 1], for a word." 
  echo "  --thresh <int>   # Pruning threshold for n-best."
  exit 1;
fi

wordlist=$1
modeldir=$2
outdir=$3

model=$modeldir/model.fst
output_lex=$outdir/lexicon.lex
mkdir -p $outdir

[ ! -f ${model:-} ] && echo "$0: File $model not found in the directory $modeldir." && exit 1
[ ! -f $wordlist ] && echo "$0: File $wordlist not found!" && exit 1
[ -z $pmass ] && [ -z $nbest ] && echo "$0: nbest or/and pmass should be specified." && exit 1;
if ! phonetisaurus=`which phonetisaurus-apply` ; then
  echo "Phonetisarus was not found !"
  echo "Go to $KALDI_ROOT/tools and execute extras/install_phonetisaurus.sh"
  exit 1
fi

cp $wordlist $outdir/wordlist.txt

# three options: 1) nbest, 2) pmass, 3) nbest+pmass,
nbest=${nbest:-20}   # if nbest is not specified, set it to 20, due to Phonetisaurus mechanism
pmass=${pmass:-1.0}  # if pmass is not specified, set it to 1.0, due to Phonetisaurus mechanism

[[ ! $nbest =~ ^[1-9][0-9]*$ ]] && echo "$0: nbest should be a positive integer." && exit 1;

echo "Applying the G2P model to wordlist $wordlist"
phonetisaurus-apply --pmass $pmass --nbest $nbest --thresh $thresh \
  --word_list $wordlist --model $model \
  --accumulate --verbose --prob \
  1>$output_lex

echo "Completed. Synthesized lexicon for new words is in $output_lex"

# Some words might have been removed or skipped during the process,
# let's check it and warn the user if so...
nlex=`cut -f 1 $output_lex | sort -u | wc -l`
nwlist=`cut -f 1 $wordlist | sort -u | wc -l`
if [ $nlex -ne $nwlist ] ; then
  failed_wordlist=$outdir/lexicon.failed
  echo "WARNING: Unable to generate pronunciation for all words. ";
  echo "WARINNG:   Wordlist: $nwlist words"
  echo "WARNING:   Lexicon : $nlex words"
  comm -13 <(cut -f 1 $output_lex | sort -u ) \
           <(cut -f 1 $wordlist | sort -u ) \
           >$failed_wordlist && echo "WARNING: The list of failed words is in $failed_wordlist"
fi
exit 0

