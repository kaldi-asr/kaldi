#!/usr/bin/env bash

# this script is to train a small, pruned n-gram backoff LM to be used for sampling
# purposes during RNNLM training.  We ue pocolm for this because it's good at pruning,
# maintained by us so we can ensure it works, and has no licensing problems.

dir=exp/rnnlm_data_prep

# To be run from the directory egs/ptb/s5.
# to be run after prepare_rnnlm_data.sh.  this will all later be refactored.

. ./path.sh
set -e
[ -z "$KALDI_ROOT" ] && echo "$0: KALDI_ROOT is not set in path.sh" && exit 1
pocolm=$KALDI_ROOT/tools/pocolm

if [ ! -f $pocolm/scripts/train_lm.py ]; then
  echo "$0: you should install pocolm.  cd to $KALDI_ROOT/tools and run extras/install_pocolm.sh."
  exit 1
fi

# version of data/text with just the things needed for pocolm.
mkdir -p data/pocolm/text
cp data/text/*.txt data/pocolm/text  # wasteful, yes...

$pocolm/scripts/validate_text_dir.py data/pocolm/text
tail -n +5 data/vocab/words.txt | awk '{print $1}' > data/pocolm/wordlist

# later could consider using min-counts.

$pocolm/scripts/train_lm.py  --wordlist=data/pocolm/wordlist --num-splits=10 --warm-start-ratio=20  \
                             data/pocolm/text 3 data/pocolm/work data/pocolm/lm

$pocolm/scripts/prune_lm_dir.py  --target-num-ngrams=100000 data/pocolm/lm data/pocolm/lm_pruned100k

$pocolm/scripts/format_arpa_lm.py data/pocolm/lm_pruned100k | gzip -c >data/pocolm/trigram_100k.arpa.gz
