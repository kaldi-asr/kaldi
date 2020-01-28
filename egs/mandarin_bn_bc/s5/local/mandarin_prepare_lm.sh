#!/usr/bin/env bash

ngram_order=4
oov_sym="<UNK>"
no_uttid="false"
prune_thres=1e-9

[ -f ./path.sh ] && . ./path.sh
. parse_options.sh || exit 1;

if [ $# != 4 ]; then
  echo "Usage: [--no-uttid] [--ngram-order] [--oov-sym] [--prune-thres] <dict-dir> <src-dir> <lm-dir> <dev-dir>"
  echo "E.g. $0 --no-uttid "true" --ngram-order 4 --oov-sym \"<UNK>\" --prune-thres "1e-9" data/local/dict data/local/train data/local/lm data/dev "
  exit 1;
fi

dict_dir=$1
local_text_dir=$2
lm_dir=$3
heldout=$4/text

# check if sri is installed or no
which ngram-count  &>/dev/null
if [[ $? == 0 ]]; then
  echo "srilm installed"
else
  echo "Please install srilm first !"
  exit 1
fi
echo "Building $ngram_order gram LM"
[ ! -d $lm_dir ] && mkdir -p $lm_dir exit 1;

if [ ! -f $lm_dir/${ngram_order}gram-mincount/lm_pruned.gz ]; then
  echo "Training LM with train text"
  [ ! -f $local_text_dir/text ] && echo "No $local_text_dir/text" && exit 1;

  # If the first column of $local_text_dir/text is uttid, we need to remove
  # them.
  if [ $no_uttid == "false" ]; then 
    awk '{i=$2;for (n=3;n<=NF;++n){i=i" "$n;}print i}' $local_text_dir/text > $lm_dir/text
  else
    cp $local_text_dir/text $lm_dir/text
  fi
  local/train_lms.sh --ngram-order $ngram_order --prune-thres $prune_thres $lm_dir $dict_dir $lm_dir $heldout
fi


