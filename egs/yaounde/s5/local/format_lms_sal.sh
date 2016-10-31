#!/bin/bash
# Prepares the test time language model(G) transducers
# (adapted from wsj/s5/local/wsj_format_data.sh)
. ./path.sh || exit 1;
src_dir=data/lang
. utils/parse_options.sh || exit 1;
set -e
if [ $# -ne 1 ]; then
    echo "Usage: $0 <lm-dir>"
    echo "e.g.: $0 data/lm"
    echo ", where:"
    echo "    <lm-dir> is the directory in which the language model is stored/downloaded"
    echo "Options:"
    echo "   --src-dir  <dir>           # source lang directory, default data/lang"
    exit 1
fi
lm_dir=$1
if [ ! -d $lm_dir ]; then
    echo "$0: expected source LM directory $lm_dir to exist"
    exit 1;
fi
if [ ! -f $src_dir/words.txt ]; then
    echo "$0: expected $src_dir/words.txt to exist."
    exit 1;
fi
tmpdir=data/local/lm_tmp.$$
trap "rm -r $tmpdir" EXIT
mkdir -p $tmpdir
test=${src_dir}_test_threegram_sal
mkdir -p $test
cp -r ${src_dir}/* $test
gunzip -c $lm_dir/lm_threegram.arpa.gz | \
    utils/find_arpa_oovs.pl \
	$test/words.txt > \
	$tmpdir/oovs_threegram.txt || exit 1
gunzip -c $lm_dir/lm_threegram.arpa.gz | \
    grep -v '<s> <s>' | \
    grep -v '</s> <s>' | \
    grep -v '</s> </s>' | \
    arpa2fst - | \
    fstprint | \
    utils/remove_oovs.pl \
	$tmpdir/oovs_threegram.txt | \
    utils/eps2disambig.pl | \
    utils/s2eps.pl | \
    fstcompile \
	--isymbols=$test/words.txt \
	--osymbols=$test/words.txt \
	--keep_isymbols=false \
	--keep_osymbols=false | \
    fstrmepsilon | \
    fstarcsort \
	--sort_type=ilabel > \
	$test/G.fst
utils/validate_lang.pl \
    --skip-determinization-check \
    $test || exit 1;
exit 0
