#!/usr/bin/env bash

# Create the LMs and graphs and decode all test sets.
# Some paths are hard coded in this script.

set -e

stage=0
nj=30
ivector_base=  # Set to empty to disable ivector
test_sets="fsh_dev nsc_ivr_test5k ted_test cv_en_indian_testdev cv_en_indian_other ted_dev"
tree_dir=
iter=final
graphs=""
decode_opts=""
suffix=""
overwrite=false
lm=            # Set the lm if you have not already created "lang_test" under you $exp dir

echo "$0 $@"  # Print the command line for logging

. ./cmd.sh
. ./path.sh
. ./utils/parse_options.sh

if [ $# != 1 ]; then
  echo "Usage: $0 [--ivector-base <ivector-base-dir>] <model-dir>"
  echo "E.g.: $0 exp/chain/tdnn1b "
  echo "Other options:"
  echo "--test-sets <list-of-test-sets>"
  echo "--tree-dir <tree-dir>"
  echo "--iter <final/integer>"
  echo "--graphs <prebuilt graphs>"
  exit 1;
fi


dir=$1
exp=$(echo $1 | cut -d'/' -f1)
if [ -z $tree_dir ]; then
  tree_dir=$(cat $dir/tree_dir) || true
  [ -z $tree_dir ] && tree_dir=$(cat $dir/src_tree_dir.txt) || true
  [ -z $tree_dir ] && tree_dir=$exp/chain/tree_sp
fi

if [ $stage -le 1 ]; then
  if [ ! -e $exp/lang_test ]; then
    utils/format_lm.sh $exp/lang $lm \
                       $exp/dict/lexicon.txt $exp/lang_test
  fi
fi

if [ -z $graphs ]; then
  if [ $stage -le 2 ]; then
      if [ ! -f $tree_dir/graph/HCLG.fst ]; then
        utils/mkgraph.sh \
          --self-loop-scale 1.0 $exp/lang_test \
          $tree_dir $tree_dir/graph || exit 1;
      fi
  fi
  graphs="$tree_dir/graph"
fi

if [ $stage -le 3 ]; then
  local/decode_all.sh --iter $iter --nj $[nj/2] --graphs "$graphs" --test-sets "$test_sets" --overwrite $overwrite \
                      --ivector-base "$ivector_base" --decode-opts "$decode_opts" --suffix "$suffix" --dir $dir
fi
