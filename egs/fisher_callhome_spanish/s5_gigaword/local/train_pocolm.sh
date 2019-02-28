#!/bin/bash

stage=-2
num_words_pocolm=110000
prune_size=1000000

. ./path.sh
. ./cmd.sh
. ./utils/parse_options.sh

set -euo pipefail

export POCOLM_ROOT=$(cd $KALDI_ROOT/tools/pocolm/; pwd -P)
export PATH=$PATH:$POCOLM_ROOT/scripts

textdir=$1
pocolm_dir=$2


if [ $stage -le -2 ]; then
    echo "****"
    echo " POCOLM experiment : Running STAGE 1 : 2-gram Pocolm general closed vocabulary model"
    echo " Will estimate the metaparams to be used as unigram weights for stage 2 ....."
    echo "****"
    if [ -e "$textdir"/unigram_weights ]; then
	rm "$textdir"/unigram_weights
    fi
    if [ -e "$pocolm_dir" ]; then
	rm -r "$pocolm_dir"
    fi
    
    bash local/pocolm_cust.sh  --num-word 0 --ngram-order 2 --pocolm-stage 1 --lm-dir "$pocolm_dir"/lm \
	 --arpa-dir "$pocolm_dir"/arpa --textdir "$textdir"

fi
    
if [ $stage -le -1 ];then
    echo "********"
    echo "POCOLM experiment : RUNNING STAGE 2 : 3gram POCOLM using unigram wts estimates in 1st stage....."
    echo "********"

    echo " " > "$pocolm_dir"/lm/work/.unigram_weights.done
    python local/get_unigram_weights_vocab.py "$pocolm_dir"/lm/0_2.pocolm/ "$textdir"/unigram_weights
    bash local/pocolm_cust.sh  --num-word "$num_words_pocolm"  --lm-dir "$pocolm_dir"/lm \
	                       --arpa-dir "$pocolm_dir"/arpa --textdir "$textdir"
    

fi


exit 0;
