#!/usr/bin/env bash
#
# Copyright  2018  David Snyder
# Apache 2.0
#
# This script downloads pre-built language models trained on the Cantab-Tedlium
# text data and Tedlium acoustic training data.  If you want to build these
# models yourself, run the script local/ted_train_lm.sh.

set -e

echo "$0: downloading Tedlium 4 gram language models (it won't re-download if it was already downloaded.)"
wget --continue http://kaldi-asr.org/models/5/4gram_small.arpa.gz -P data/local/local_lm/data/arpa || exit 1
wget --continue http://kaldi-asr.org/models/5/4gram_big.arpa.gz -P data/local/local_lm/data/arpa || exit 1

exit 0

