#!/usr/bin/env bash
# 2020 author Jiayu DU
# Apache 2.0

# This script reads in an Arpa format language model, and converts it into the
# KenLM format language model.

[ -f path.sh ] && . ./path.sh;

# begin configuration section
kenlm_opts="" # e.g. "-q 8 -b 8" for 8bits quantization
model_type="trie" # "trie" or "probing". trie is smaller, probing is faster.
# end configuration section

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

if ! which build_binary >& /dev/null ; then
  echo "$0: cannot find KenLM's build_binary tool,"
  echo "check kenlm installation (tools/extras/install_kenlm_query_only.sh)."
  exit 1
fi

mkdir -p $(dirname $kenlm)
build_binary  $kenlm_opts  $model_type  $arpa_lm  $kenlm

echo "$0: Successfully built arpa into kenlm format: $kenlm"
exit 0
