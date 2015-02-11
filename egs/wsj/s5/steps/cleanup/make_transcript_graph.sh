#!/bin/bash

# Copyright 2014  Guoguo Chen
# Apache 2.0

# Begin configuration section.
tscale=1.0      # transition scale.
loopscale=0.1   # scale for self-loops.
cleanup=true
# End configuration section.

set -e

echo "$0 $@"

[ -f ./path.sh ] && . ./path.sh
. parse_options.sh || exit 1;

if [ $# -ne 4 ]; then
  echo "This script builds one decoding graph for each transcript in the given"
  echo "<text> file."
  echo ""
  echo "Usage: $0 [options] <text> <lang-dir> <model-dir> <graph-dir>"
  echo "Options:"
  echo "    --lm-order              # order of n-gram language model"
  echo "    --lm-options            # options for ngram-count in SRILM tool"
  exit 1;
fi

text=$1
lang=$2
model_dir=$3
graph_dir=$4

for f in $lang/L_disambig.fst $lang/words.txt $lang/oov.int \
  $model_dir/final.mdl $model_dir/tree; do
  if [ ! -f $f ]; then
    echo "$0: expected $f to exist"
    exit 1;
  fi
done

mkdir -p $graph_dir/sub_graphs

# Maps OOV words to the oov symbol.
oov=`cat $lang/oov.int`

N=`tree-info --print-args=false $model_dir/tree |\
  grep "context-width" | awk '{print $NF}'`
P=`tree-info --print-args=false $model_dir/tree |\
  grep "central-position" | awk '{print $NF}'`

# Loops over all utterances.
rm -rf $graph_dir/sub_graphs/HCLG.fsts.scp
while read line; do
  uttid=`echo $line | cut -d ' ' -f 1`
  words=`echo $line | cut -d ' ' -f 2-`

  echo "$0: processing utterance $uttid."

  wdir=$graph_dir/sub_graphs/$uttid
  mkdir -p $wdir
  echo $words > $wdir/text

  cat $wdir/text | utils/sym2int.pl --map-oov $oov -f 1- $lang/words.txt | \
    utils/make_unigram_grammar.pl | fstcompile |\
    fstarcsort --sort_type=ilabel > $wdir/G.fst || exit 1;
  fstisstochastic $wdir/G.fst || echo "$0: $uttid/G.fst not stochastic."

  # Builds LG.fst
  fsttablecompose $lang/L_disambig.fst $wdir/G.fst |\
    fstdeterminizestar --use-log=true | fstminimizeencoded |\
    fstarcsort --sort_type=ilabel > $wdir/LG.fst || exit 1;
  fstisstochastic $wdir/LG.fst || echo "$0: $uttid/LG.fst not stochastic."

  # Builds CLG.fst
  clg=$wdir/CLG_${N}_${P}.fst
  fstcomposecontext --context-size=$N --central-position=$P \
    --read-disambig-syms=$lang/phones/disambig.int \
    --write-disambig-syms=$wdir/disambig_ilabels_${N}_${P}.int \
    $wdir/ilabels_${N}_${P} < $wdir/LG.fst | fstdeterminize > $wdir/CLG.fst
  fstisstochastic $wdir/CLG.fst  || echo "$0: $uttid/CLG.fst not stochastic."

  make-h-transducer --disambig-syms-out=$wdir/disambig_tid.int \
    --transition-scale=$tscale $wdir/ilabels_${N}_${P} \
    $model_dir/tree $model_dir/final.mdl > $wdir/Ha.fst 

  # Builds HCLGa.fst
  fsttablecompose $wdir/Ha.fst $wdir/CLG.fst | \
    fstdeterminizestar --use-log=true | \
    fstrmsymbols $wdir/disambig_tid.int | fstrmepslocal | \
    fstminimizeencoded > $wdir/HCLGa.fst
  fstisstochastic $wdir/HCLGa.fst ||\
    echo "$0: $uttid/HCLGa.fst is not stochastic"
  
  add-self-loops --self-loop-scale=$loopscale --reorder=true \
    $model_dir/final.mdl < $wdir/HCLGa.fst > $wdir/HCLG.fst
  
  if [ $tscale == 1.0 -a $loopscale == 1.0 ]; then
    fstisstochastic $wdir/HCLG.fst ||\
      echo "$0: $uttid/HCLG.fst is not stochastic."
  fi

  echo "$uttid $wdir/HCLG.fst" >> $graph_dir/sub_graphs/HCLG.fsts.scp
  echo
done < $text

# Copies files from lang directory.
mkdir -p $graph_dir
cp -r $lang/* $graph_dir

am-info --print-args=false $model_dir/final.mdl |\
 grep pdfs | awk '{print $NF}' > $graph_dir/num_pdfs

# Creates the graph table.
fstcopy scp:$graph_dir/sub_graphs/HCLG.fsts.scp \
  "ark,scp:$graph_dir/HCLG.fsts,$graph_dir/HCLG.fsts.scp"

if $cleanup; then
  rm -rf $graph_dir/sub_graphs
fi

exit 0;
