#!/bin/bash

# Copyright 2014  Guoguo Chen
# Apache 2.0

# Begin configuration section.
nj=4
cmd=run.pl
tscale=1.0      # transition scale.
loopscale=0.1   # scale for self-loops.
cleanup=true
# End configuration section.

set -e

echo "$0 $@"

[ -f ./path.sh ] && . ./path.sh
. parse_options.sh || exit 1;

if [ $# -ne 4 ]; then
  echo "This script is a wrapper of steps/cleanup/make_transcript_graph.sh. In"
  echo "the segmentation case graphs are created for the original transcript"
  echo "(the long transcript before split), therefore we have to duplicate the"
  echo "graphs for the new utterances. We do this in the scp file so that we"
  echo "can avoid storing the duplicate graphs on the disk."
  echo ""
  echo "Usage: $0 [options] <data-dir> <lang-dir> <model-dir> <graph-dir>"
  echo "Options:"
  echo "    --lm-order              # order of n-gram language model"
  echo "    --lm-options            # options for ngram-count in SRILM tool"
  exit 1;
fi

data=$1
lang=$2
model_dir=$3
graph_dir=$4

for f in $data/text.orig $data/orig2utt $lang/L_disambig.fst \
  $lang/words.txt $lang/oov.int $model_dir/final.mdl $model_dir/tree; do
  if [ ! -f $f ]; then
    echo "$0: expected $f to exist"
    exit 1;
  fi
done

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

$cmd JOB=1:$nj $graph_dir/log/make_transcript_graph.JOB.log \
  steps/cleanup/make_transcript_graph.sh --cleanup $cleanup \
  --tscale $tscale --loopscale $loopscale \
  $graph_dir/split$nj/JOB/text $lang \
  $model_dir $graph_dir/split$nj/JOB || exit 1;

# Copies files from lang directory.
mkdir -p $graph_dir
cp -r $lang/* $graph_dir

am-info --print-args=false $model_dir/final.mdl |\
 grep pdfs | awk '{print $NF}' > $graph_dir/num_pdfs

# Creates the graph table.
cat $graph_dir/split$nj/*/HCLG.fsts.scp > $graph_dir/split$nj/HCLG.fsts.scp
fstcopy scp:$graph_dir/split$nj/HCLG.fsts.scp \
  "ark,scp:$graph_dir/HCLG.fsts,$graph_dir/tmp.HCLG.fsts.scp"

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
    @col == 2 || die "Error: bad line $_\n";
    defined($scp{$col[0]}) ||
      die "Error: $col[0] not defined in original scp file\n";
    print "$col[1] $scp{$col[0]}\n"
  }' $data/orig2utt > $graph_dir/HCLG.fsts.scp
rm -rf $graph_dir/tmp.HCLG.fsts.scp

if $cleanup; then
  rm -rf $graph_dir/split$nj
fi

exit 0;
