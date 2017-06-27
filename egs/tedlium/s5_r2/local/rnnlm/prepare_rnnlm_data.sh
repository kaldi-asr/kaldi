#!/bin/bash

# To be run from the egs/ directory.

. path.sh

set -e -o pipefail -u

export LC_ALL=C
export PYTHONIOENCODING='utf-8'

# it should contain things like
# foo.txt, bar.txt, and dev.txt (dev.txt is a special filename that's
# obligatory).
dir=data/rnnlm/

# validata data dir
rnnlm/validate_data_dir.py $dir/data/

# get unigram counts
rnnlm/get_unigram_counts.sh $dir/data/

# get vocab
rnnlm/get_vocab.py $dir/data > $dir/vocab/words.txt

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
mkdir -p exp/rnnlm/
cat > exp/rnnlm/data_weights.txt <<EOF
ted 1   1.0
EOF
