#!/bin/bash

# steps/make_phone_graph.sh data/train_100k_nodup/ data/lang exp/tri2_ali_100k_nodup/ exp/tri2

# Copyright 2013  Johns Hopkins University (Author: Daniel Povey).  Apache 2.0.

# This script makes a phone-based LM, without smoothing to unigram, that 
# is to be used for segmentation, and uses that together with a model to
# make a decoding graph.
# Uses SRILM.

# Begin configuration section.
stage=0
cmd=run.pl
iter=final  # use $iter.mdl from $model_dir
tree=tree
tscale=1.0 # transition scale.
loopscale=0.1 # scale for self-loops.
# End configuration section.

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

if [ $# -ne 3 ]; then
  echo "Usage: $0  [options] <lang-dir> <model-dir> <dir>"
  echo " e.g.: $0 exp/vad_dev/lang exp/vad_dev exp/vad_dev/graph"
  echo "Makes the graph in \$dir, corresponding to the model in \$model_dir"
  exit 1;
fi

lang=$1
model=$2/$iter.mdl
tree=$2/$tree
dir=$3

for f in $lang/G.fst $model $tree; do
  if [ ! -f $f ]; then
    echo "$0: expected $f to exist"
    exit 1;
  fi
done

mkdir -p $dir $lang/tmp

clg=$lang/tmp/CLG.fst

if [[ ! -s $clg || $clg -ot $lang/G.fst ]]; then
  echo "$0: creating CLG."

  fstcomposecontext --context-size=1 --central-position=0 \
    $lang/tmp/ilabels < $lang/G.fst | \
    fstarcsort --sort_type=ilabel > $clg
  fstisstochastic $clg || echo "[info]: CLG not stochastic."
fi

if [[ ! -s $dir/Ha.fst || $dir/Ha.fst -ot $model || $dir/Ha.fst -ot $lang/tmp/ilabels ]]; then
  make-h-transducer --disambig-syms-out=$dir/disambig_tid.int \
    --transition-scale=$tscale $lang/tmp/ilabels $tree $model \
    > $dir/Ha.fst || exit 1;
fi

if [[ ! -s $dir/HCLGa.fst || $dir/HCLGa.fst -ot $dir/Ha.fst || $dir/HCLGa.fst -ot $clg ]]; then
  fsttablecompose $dir/Ha.fst $clg | fstdeterminizestar --use-log=true \
    | fstrmsymbols $dir/disambig_tid.int | fstrmepslocal | \
    fstminimizeencoded > $dir/HCLGa.fst || exit 1;
  fstisstochastic $dir/HCLGa.fst || echo "HCLGa is not stochastic"
fi

if [[ ! -s $dir/HCLG.fst || $dir/HCLG.fst -ot $dir/HCLGa.fst ]]; then
  add-self-loops --self-loop-scale=$loopscale --reorder=true \
    $model < $dir/HCLGa.fst > $dir/HCLG.fst || exit 1;

  if [ $tscale == 1.0 -a $loopscale == 1.0 ]; then 
    # No point doing this test if transition-scale not 1, as it is bound to fail.
    fstisstochastic $dir/HCLG.fst || echo "[info]: final HCLG is not stochastic."
  fi
fi

# keep a copy of the lexicon and a list of silence phones with HCLG...
# this means we can decode without reference to the $lang directory.

cp $lang/words.txt $dir/ || exit 1;
mkdir -p $dir/phones
cp $lang/phones/word_boundary.* $dir/phones/ 2>/dev/null # might be needed for ctm scoring,
cp $lang/phones/align_lexicon.* $dir/phones/ 2>/dev/null # might be needed for ctm scoring,
  # but ignore the error if it's not there.

cp $lang/phones/disambig.{txt,int} $dir/phones/ 2> /dev/null
cp $lang/phones/silence.csl $dir/phones/ || exit 1;
cp $lang/phones.txt $dir/ 2> /dev/null # ignore the error if it's not there.

# to make const fst:
# fstconvert --fst_type=const $dir/HCLG.fst $dir/HCLG_c.fst
am-info --print-args=false $model | grep pdfs | awk '{print $NF}' > $dir/num_pdfs
