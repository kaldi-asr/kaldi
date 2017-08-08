#!/bin/bash

# To be run from the directory egs/ptb/s5.

# . path.sh
set -e
# export LC_ALL=C
# export PYTHONIOENCODING='utf-8'

# it should contain things like
# foo.txt, bar.txt, and dev.txt (dev.txt is a special filename that's obligatory).
mkdir -p data/text
cp data/ptb/ptb.txt  data/text/
cp data/ptb/dev.txt  data/text/


# validata data dir
rnnlm/validate_data_dir.py data/text

# get unigram counts
rnnlm/get_unigram_counts.sh data/text

# get vocab
mkdir -p data/vocab
rnnlm/get_vocab.py data/text > data/vocab/words.txt


dir=exp/rnnlm_data_prep
mkdir -p $dir

# Choose weighting and multiplicity of data.
# The following choices would mean that data-source 'foo'
# is repeated once per epoch and has a weight of 0.5 in the
# objective function when training, and data-source 'bar' is repeated twice
# per epoch and has a data -weight of 1.5.
# There is no contraint that the average of the data weights equal one.
# Note: if a data-source has zero multiplicity, it just means you are ignoring
# it; but you must include all data-sources.
#cat > exp/foo/data_weights.txt <<EOF
#foo 1   0.5
#bar 2   1.5
#baz 0   0.0
#EOF
cat > $dir/data_weights.txt <<EOF
ptb   1   1.0
EOF

# get unigram probs; this also validates the 'data-weights' file.
rnnlm/get_unigram_probs.py --vocab-file=data/vocab/words.txt \
                           --data-weights-file=$dir/data_weights.txt \
                           data/text >$dir/unigram_probs.txt

# choose features
rnnlm/choose_features.py --unigram-probs=$dir/unigram_probs.txt \
                         --unigram-scale=0.1 \
                         data/vocab/words.txt > $dir/features.txt
# validate features
rnnlm/validate_features.py $dir/features.txt

# make features for word
rnnlm/make_word_features.py --unigram-probs=$dir/unigram_probs.txt \
                        data/vocab/words.txt $dir/features.txt \
                         > $dir/word_feats.txt

# validate word features
rnnlm/validate_word_features.py --features-file $dir/features.txt \
                                $dir/word_feats.txt
