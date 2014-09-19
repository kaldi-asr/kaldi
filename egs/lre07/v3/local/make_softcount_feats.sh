#!/bin/bash

. cmd.sh
. path.sh
set -e

cmd="$train_cmd -l mem_free=2G,ram_free=2G"
base=exp/phonotactics
order=3
num_feats=5000

mkdir -p $base/$order
mkdir -p $base/log

$cmd JOB=1:25 $base/log/test_ngram_counts.JOB.log \
  gunzip -c exp/tri5a/lre07_basis_fmllr_special/lat.JOB.gz \| \
  lattice-expand-ngram --n=$order ark:- ark:- \| \
  lattice-to-ngram-counts --n=$order --acoustic-scale=0.075 --eos-symbol=100 \
  ark:- $base/$order/test_counts.JOB || exit 1;

echo "Created expected test ngram counts";

$cmd JOB=1:50 $base/log/train_ngram_counts.JOB.log \
  gunzip -c exp/tri5a/train_basis_fmllr_special/lat.JOB.gz \| \
  lattice-expand-ngram --n=$order ark:- ark:- \| \
  lattice-to-ngram-counts --n=$order --acoustic-scale=0.075 --eos-symbol=100 \
  ark:- $base/$order/train_counts.JOB || exit 1;

echo "Created expected train ngram counts";

cat $base/$order/test_counts.* | gzip -c > $base/test_counts.gz
echo "Accumulated test ngram counts.";
cat $base/$order/train_counts.* | gzip -c > $base/train_counts.gz
echo "Accumulated train ngram counts.";

rm $base/$order/test_counts.*
rm $base/$order/train_counts.*

local/get_top_ngrams_filtered.pl <(gunzip -c $base/train_counts.gz) | head -n $num_feats > $base/ngram_table
local/get_ngram_count.pl <(gunzip -c $base/train_counts.gz) $base/ngram_table 1 > $base/train_feats_unnorm.txt
local/get_ngram_count.pl <(gunzip -c $base/test_counts.gz) $base/ngram_table 0 > $base/test_feats_unnorm.txt

echo "Got unnormalized feats.";

local/apply_freq.pl $base/train_feats_unnorm.txt $base/ngram_table $base/train_feats.txt
local/apply_freq.pl $base/test_feats_unnorm.txt $base/ngram_table $base/test_feats.txt

echo "Normalized feats.";

copy-vector ark,t:$base/test_feats.txt ark,scp:$base/test_feats.ark,$base/test_feats.scp
copy-vector ark,t:$base/train_feats.txt ark,scp:$base/train_feats.ark,$base/train_feats.scp

rm $base/test_feats*txt
rm $base/train_feats*txt

utils/filter_scp.pl -f 0 $base/train_feats.scp data/train/utt2lang > $base/utt2lang_train
utils/filter_scp.pl -f 0 $base/test_feats.scp data/lre07/utt2lang > $base/utt2lang_test

utils/filter_scp.pl -f 0 data/lre07/30sec $base/utt2lang_test > $base/utt2lang_test30s
utils/filter_scp.pl -f 0 data/lre07/30sec $base/test_feats.scp > $base/test_feats30s.scp
