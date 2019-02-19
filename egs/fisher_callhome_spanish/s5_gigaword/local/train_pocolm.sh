#!/bin/bash

stage=-2
num_words_pocolm=110000
prune_size=1000000

. ./path_venv.sh
. ./cmd.sh
. ./utils/parse_options.sh

set -euo pipefail

export POCOLM_ROOT=$(cd $KALDI_ROOT/tools/pocolm/; pwd -P)
export PATH=$PATH:$POCOLM_ROOT/scripts

textdir=$1
pocolm_dir=$2


if [ $stage -le -2 ];then
    if [ -e "$textdir"/unigram_weights ]; then
	rm "$textdir"/unigram_weights
    fi

    if [ -e "$pocolm_dir" ]; then
	rm -r "$pocolm_dir"
    fi

    bash local/pocolm_cust.sh  --num-word "$num_words_pocolm" --lm-dir "$pocolm_dir"/lm \
	                       --arpa-dir "$pocolm_dir"/arpa --textdir "$textdir"
fi

if [ $stage -le -1 ];then
  prune_lm_dir.py --target-num-ngrams=${prune_size} --max-memory=8G "$pocolm_dir"/lm/"$num_words_pocolm"_3.pocolm "$pocolm_dir"/lm/"$num_words_pocolm"_3.pocolm_pruned
  format_arpa_lm.py --max-memory=8G "$pocolm_dir"/lm/"$num_words_pocolm"_3.pocolm_pruned | gzip -c > "$pocolm_dir"/arpa/"$num_words_pocolm"_3.pocolm_pruned_${prune_size}.arpa.gz
fi


exit 0;
