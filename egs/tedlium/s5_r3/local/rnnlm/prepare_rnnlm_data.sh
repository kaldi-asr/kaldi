#!/usr/bin/env bash

# To be run from the egs/ directory.

. path.sh

set -e -o pipefail -u

# it should contain things like
# foo.txt, bar.txt, and dev.txt (dev.txt is a special filename that's
# obligatory).
data_dir=data/rnnlm
dir=exp/rnnlm/
mkdir -p $dir

# validata data dir
rnnlm/validate_data_dir.py $data_dir/data/

# get unigram counts
rnnlm/ensure_counts_present.sh $data_dir/data/

# get vocab
mkdir -p $data_dir/vocab
rnnlm/get_vocab.py $data_dir/data > $data_dir/vocab/words.txt

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
ted 1   1.0
EOF

# get unigram probs
rnnlm/get_unigram_probs.py --vocab-file=$data_dir/vocab/words.txt \
                           --data-weights-file=$dir/data_weights.txt \
                           $data_dir/data > $dir/unigram_probs.txt

# choose features
rnnlm/choose_features.py --unigram-probs=$dir/unigram_probs.txt \
                         $data_dir/vocab/words.txt > $dir/features.txt
# validate features
rnnlm/validate_features.py $dir/features.txt

# make features for word
rnnlm/make_word_features.py --unigram-probs=$dir/unigram_probs.txt \
                         $data_dir/vocab/words.txt $dir/features.txt \
                         > $dir/word_feats.txt

# validate word features
rnnlm/validate_word_features.py --features-file $dir/features.txt \
                                $dir/word_feats.txt
