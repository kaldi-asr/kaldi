#!/usr/bin/env bash

# To be run from the directory egs/ptb/s5.

# . path.sh
set -e


# it should contain things like
# foo.txt, bar.txt, and dev.txt (dev.txt is a special filename that's obligatory).
mkdir -p data/text
cp data/ptb/ptb.txt  data/text/
cp data/ptb/dev.txt  data/text/

# validata data dir
rnnlm/validate_text_dir.py data/text

# get unigram counts; these are used by rnnlm/get_vocab.py.
rnnlm/ensure_counts_present.sh data/text

# get vocab
mkdir -p data/vocab
rnnlm/get_vocab.py data/text > data/vocab/words.txt
