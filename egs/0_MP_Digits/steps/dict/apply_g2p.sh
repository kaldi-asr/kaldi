#!/bin/bash
# Copyright 2014  Johns Hopkins University (Author: Yenda Trmal)
# Copyright 2016  Xiaohui Zhang
# Apache 2.0

# Begin configuration section.  
stage=0
encoding='utf-8'
var_counts=3  #Generate upto N variants
var_mass=0.9  #Generate so many variants to produce 90 % of the prob mass
cmd=run.pl
nj=10          #Split the task into several parallel, to speedup things
model=
# End configuration section.

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

set -u
set -e

if [ $# != 3 ]; then
   echo "Usage: $0 [options] <word-list> <g2p-model-dir> <output-dir>"
   echo "... where <word-list> is a list of words whose pronunciation is to be generated"
   echo "          <g2p-model-dir> is a directory used as a target during training of G2P"
   echo "          <output-dir> is the directory where the output lexicon should be stored"
   echo "e.g.: $0 oov_words exp/g2p exp/g2p/oov_lex"
   echo ""
   echo "main options (for others, see top of script file)"
   echo "  --nj <int>                                    # How many tasks should be spawn (to speedup things)"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   exit 1;
fi

wordlist=$1
modeldir=$2
output=$3


mkdir -p $output/log

model=$modeldir/g2p.model.final
[ ! -f ${model:-} ] && echo "File $model not found in the directory $modeldir." && exit 1
#[ ! -x $wordlist ] && echo "File $wordlist not found!" && exit 1

cp $wordlist $output/wordlist.txt

if ! g2p=`which g2p.py` ; then
  echo "The Sequitur was not found !"
  echo "Go to $KALDI_ROOT/tools and execute extras/install_sequitur.sh"
  exit 1
fi

echo "Applying the G2P model to wordlist $wordlist"

if [ $stage -le 0 ]; then
  $cmd JOBS=1:$nj $output/log/apply.JOBS.log \
    split -n l/JOBS/$nj $output/wordlist.txt \| \
    g2p.py -V $var_mass --variants-number $var_counts --encoding $encoding \
      --model $modeldir/g2p.model.final --apply - \
    \> $output/output.JOBS
fi
cat $output/output.* > $output/output

# Remap the words from output file back to the original casing
# Conversion of some of thems might have failed, so we have to be careful
# and use the transform_map file we generated beforehand
# Also, because the sequitur output is not readily usable as lexicon (it adds 
# one more column with ordering of the pron. variants) convert it into the proper lexicon form
output_lex=$output/lexicon.lex

# Just convert it to a proper lexicon format
cut -f 1,3,4 $output/output > $output_lex

# Some words might have been removed or skipped during the process,
# let's check it and warn the user if so...
nlex=`cut -f 1 $output_lex | sort -u | wc -l`
nwlist=`cut -f 1 $output/wordlist.txt | sort -u | wc -l`
if [ $nlex -ne $nwlist ] ; then
  echo "WARNING: Unable to generate pronunciation for all words. ";
  echo "WARINNG:   Wordlist: $nwlist words"
  echo "WARNING:   Lexicon : $nlex words"
  echo "WARNING:Diff example: "
  diff <(cut -f 1 $output_lex | sort -u ) \
       <(cut -f 1 $output/wordlist.txt | sort -u ) || true
fi
exit 0
