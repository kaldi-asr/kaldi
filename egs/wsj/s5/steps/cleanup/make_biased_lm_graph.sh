#!/bin/bash
# Copyright 2012-2014  Johns Hopkins University (Author: Daniel Povey)
#           2016       Vimal Manohar
# Apache 2.0

# Computes training alignments using a model with delta or
# LDA+MLLT features.  This version, rather than just using the
# text to align, computes mini-language models (unigram) from the text
# and a few common words in the LM.

set -u
set -o pipefail
set -e

# Begin configuration section.
nj=4
cmd=run.pl
scale_opts="--transition-scale=1.0 --self-loop-scale=0.1"
top_n_words=100 # Number of common words that we compile into each graph (most frequent
                # in $data/text.orig.
stage=-1
cleanup=true

ngram_order=2
top_words_interpolation_weight=0.1    # Interpolation weight for top-words unigram
biased_unigram_interpolation_weight=0.1 # Interpolation weight for biased unigram
biased_bigram_interpolation_weight=0.9  # Interpolation weight for biased bigram

# End configuration options.

echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh # source the path.
. parse_options.sh || exit 1;

if [ $# != 4 ]; then
   echo "usage: $0 <data-dir> <lang-dir> <src-dir> <graph-dir>"
   echo "e.g.:  $0 data/train data/lang exp/tri1 exp/tri1/graph_train"
   echo "main options (for others, see top of script file)"
   echo "  --config <config-file>                           # config containing options"
   echo "  --nj <nj>                                        # number of parallel jobs"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   exit 1;
fi

data=$1
lang=$2
model_dir=$3
graph_dir=$4

if [ ! -f $data/text.orig ] || [ ! -f $data/orig2utt ]; then
  echo "$0: This script expects a $data/text.orig which may be a copy of $data/text or something different such as a recording-specific text along with an associated mapping back to the original utterances"
  exit 1
fi

for f in $lang/oov.int $model_dir/tree $model_dir/final.mdl \
    $lang/L_disambig.fst $lang/phones/disambig.int; do
  [ ! -f $f ] && echo "$0: expected file $f to exist" && exit 1;
done

oov=`cat $lang/oov.int` || exit 1;
mkdir -p $graph_dir/log

if [ $stage -le 0 ]; then
  utils/sym2int.pl --map-oov $oov -f 2- $lang/words.txt <$data/text.orig | \
    awk '{for(x=2;x<=NF;x++) print $x;}' | sort | uniq -c | \
    sort -rn > $graph_dir/word_counts.int || exit 1;
  num_words=$(awk '{x+=$1} END{print x}' < $graph_dir/word_counts.int) || exit 1;
  # print top-n words with their unigram probabilities.

  head -n $top_n_words $graph_dir/word_counts.int | awk -v tot=$num_words '{print $1/tot, $2;}' >$graph_dir/top_words.int
  utils/int2sym.pl -f 2 $lang/words.txt <$graph_dir/top_words.int >$graph_dir/top_words.txt
fi

backoff_arc=`cat $lang/words.txt | grep "#0" | awk '{print $2}'`

max_int=`awk 'BEGIN{i=-1000} {if ($2 > i) i = $2} END{print i}' $lang/words.txt`

# Creates one graph for each transcript. We parallelize the process a little
# bit.
num_lines=`cat $data/text.orig | wc -l`
if [ $nj -gt $num_lines ]; then
  nj=$num_lines
  echo "$0: Too many number of jobs, using $nj instead"
fi

mkdir -p $graph_dir/split$nj
mkdir -p $graph_dir/log
 
split_texts=""
for n in $(seq $nj); do
  mkdir -p $graph_dir/split$nj/$n
  split_texts="$split_texts $graph_dir/split$nj/$n/text"
done
utils/split_scp.pl $data/text.orig $split_texts

if [ $stage -le 1 ]; then
  echo "$0: decoding $data using utterance-specific decoding graphs using model from $model_dir, output in $graph_dir"

  $cmd JOB=1:$nj $graph_dir/log/compile_decoding_graphs.JOB.log \
    utils/sym2int.pl --map-oov $oov -f 2- $lang/words.txt $graph_dir/split$nj/JOB/text \| \
    python steps/cleanup/make_utterance_fsts.py --top-words.list=$graph_dir/top_words.int \
    --top-words.interpolation-weight=$top_words_interpolation_weight \
    --ngram.unigram-interpolation-weight=$biased_unigram_interpolation_weight \
    --ngram.bigram-interpolation-weight=$biased_bigram_interpolation_weight \
    --ngram.order=$ngram_order --oov=$oov --backoff-arc-id=$backoff_arc \
    - - \| \
    compile-train-graphs-fsts $scale_opts --read-disambig-syms=$lang/phones/disambig.int \
    $model_dir/tree $model_dir/final.mdl $lang/L_disambig.fst ark:- \
    ark,scp:$graph_dir/split$nj/JOB/HCLG.fsts.ark,$graph_dir/split$nj/JOB/HCLG.fsts.scp || exit 1
fi

# Copies files from lang directory.
mkdir -p $graph_dir
cp -r $lang/* $graph_dir

am-info --print-args=false $model_dir/final.mdl |\
 grep pdfs | awk '{print $NF}' > $graph_dir/num_pdfs

# Creates the graph table.
for n in `seq $nj`; do
  cat $graph_dir/split$nj/$n/HCLG.fsts.scp 
done > $graph_dir/tmp.HCLG.fsts.scp

# The graphs we created above were indexed by the old utterance id. We have to
# duplicate them for the new utterance id. We do this in the scp file so we do
# not have to store the duplicated graphs on the disk.
cat $graph_dir/tmp.HCLG.fsts.scp | perl -e '
  open(O2U, "<$ARGV[0]") || die "Error: fail to open $ARGV[0]\n";
  while (<STDIN>) {
    chomp;
    @col = split;
    @col == 2 || die "Error: bad line $_\n";
    $scp{$col[0]} = $col[1];
  }
  while (<O2U>) {
    chomp;
    @col = split;
    @col >= 2 || die "Error: bad line $_\n";
    defined($scp{$col[0]}) ||
      die "Error: $col[0] not defined in original scp file\n";
    for ($i = 1; $i < @col; $i += 1) {
      print "$col[$i] $scp{$col[0]}\n"
    }
  }' $data/orig2utt > $graph_dir/HCLG.fsts.scp
rm $graph_dir/tmp.HCLG.fsts.scp

exit 0;
