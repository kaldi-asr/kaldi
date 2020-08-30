#!/usr/bin/env bash

# This script reads in an Arpa format language model, and converts it into the
# KenLM format language model.

# begin configuration section
kenlm_opts="" # e.g. "-q 8 -b 8" for 8bits quantization
model_type="trie" # eithor "trie" or "probing". trie is smaller, probing is faster.
# end configuration section

[ -f path.sh ] && . ./path.sh;

. utils/parse_options.sh

if [ $# != 2 ]; then
  echo "Usage: "
  echo "  $0 [options] <arpa-lm-path> <kenlm-path>"
  echo "e.g.:"
  echo "  $0 data/local/lm/4gram.arpa data/lang_test/G.trie"
  echo "Options:"
  echo "  --model-type can be either \"trie\" or \"probing\""
  echo "  --kenlm-opts directly pass through to kenlm"
  echo "    e.g. for 8bits quantization, feed \"-q 8 -b 8\""
  exit 1;
fi

export LC_ALL=C

arpa_lm=$1
kenlm=$2

# TODO: first we should check if kenlm is correctly installed
if [ ! "kenlm is correctly installed && successfully found compilation tool *build_binary*" ]; then
  echo "cannot find build_binary tool, please check your kenlm installation."
  exit 1
fi

mkdir -p `dirname $kenlm`
build_binary $kenlm_opts $arpa_lm $model_type $kenlm

echo "$0: Successfully built arpa into kenlm format: $kenlm"
exit 0;
