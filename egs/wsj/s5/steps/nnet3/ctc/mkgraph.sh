#!/bin/bash
#
# Copyright    2015 Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0

# This script creates a decoding graph CTC.fst for use with CTC decoding.  The
# process of creating this is simpler than for normal decoding, because in our
# version of CTC (technically, CCTC) there is no phone topology to take into
# account, and the phonetic context is only left-context.

# The output is a Finite State Transducer that has word-ids on the output, and
# CTC graph-labels on the input.

# Begin configuration section
phone_lm_weight=0.0
# End configuration section

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;


if [ $# != 3 ]; then
   echo "Usage: steps/nnet3/ctc/mkgraph.sh [options] <lang-dir> <model-dir> <graphdir>"
   echo "e.g.: steps/nnet3/ctc/mkgraph.sh data/lang exp/ctc/nnet_tdnn_a/ exp/ctc/nnet_tdnn_a/graph"
   echo "Options:"
   echo "   --phone-lm-weight  <phone-language-model-weight>  # Default 0.0.  Weight on the phone  "
   echo "                                                     # LM that CCTC was trained with."
   echo "                                                     # Should be >= 0.0 and <= 1.0."
   exit 1;
fi


if [ -f path.sh ]; then . ./path.sh; fi

lang=$1
tree=$2/tree
model=$2/final.mdl
dir=$3

mkdir -p $dir

# If $lang/tmp/LG.fst does not exist or is older than its sources, make it...
# (note: the [[ ]] brackets make the || type operators work (inside [ ], we
# would have to use -o instead),  -f means file exists, and -ot means older than).

required="$lang/L_disambig.fst $lang/G.fst $lang/phones.txt $lang/words.txt $lang/phones/silence.csl $lang/phones/disambig.int $model $tree"
for f in $required; do
  [ ! -f $f ] && echo "mkgraph.sh: expected $f to exist" && exit 1;
done

mkdir -p $lang/tmp
# Note: [[ ]] is like [ ] but enables certain extra constructs, e.g. || in
# place of -o
if [[ ! -s $lang/tmp/LG.fst || $lang/tmp/LG.fst -ot $lang/G.fst || \
      $lang/tmp/LG.fst -ot $lang/L_disambig.fst ]]; then
  fsttablecompose $lang/L_disambig.fst $lang/G.fst | fstdeterminizestar --use-log=true | \
    fstminimizeencoded | fstarcsort --sort_type=ilabel > $lang/tmp/LG.fst || exit 1;
  fstisstochastic $lang/tmp/LG.fst || echo "[info]: LG not stochastic."
fi

ctc=$dir/CTC.fst
if [[ ! -s $ctc || $ctc -ot $model ]]; then
  fstrmsymbols $lang/phones/disambig.int $lang/tmp/LG.fst | \
    ctc-make-decoding-graph --phone-lm-weight=$phone_lm_weight $model - $ctc || exit 1;
fi

cp $lang/words.txt $dir/ || exit 1;
mkdir -p $dir/phones
cp $lang/phones/word_boundary.* $dir/phones/ 2>/dev/null # might be needed for ctm scoring,
cp $lang/phones/align_lexicon.* $dir/phones/ 2>/dev/null # might be needed for ctm scoring,
  # but ignore the error if it's not there.

cp $lang/phones/disambig.{txt,int} $dir/phones/ 2> /dev/null
cp $lang/phones/silence.csl $dir/phones/ || exit 1;
cp $lang/phones.txt $dir/ 2> /dev/null # ignore the error if it's not there.

exit 0;

