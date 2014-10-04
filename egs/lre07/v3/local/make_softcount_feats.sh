#!/bin/bash
# Copyright 2014 Telepoint Global Hosting Service, LLC. (Author: David Snyder)
# See ../../COPYING for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.

. cmd.sh
. path.sh
set -e

cmd="$train_cmd -l mem_free=2G,ram_free=2G"
base=exp/phonotactics
order=3
num_feats=5000

mkdir -p $base/$order
mkdir -p $base/log

# Create soft n-gram counts for LRE07 and training data. 
$cmd JOB=1:25 $base/log/test_ngram_counts.JOB.log \
  gunzip -c exp/tri5a/lre07_basis_fmllr/lat.JOB.gz \| \
  lattice-expand-ngram --n=$order ark:- ark:- \| \
  lattice-to-ngram-counts --n=$order --acoustic-scale=0.075 --eos-symbol=100 \
  ark:- $base/$order/test_counts.JOB || exit 1;

$cmd JOB=1:50 $base/log/train_ngram_counts.JOB.log \
  gunzip -c exp/tri5a/train_basis_fmllr/lat.JOB.gz \| \
  lattice-expand-ngram --n=$order ark:- ark:- \| \
  lattice-to-ngram-counts --n=$order --acoustic-scale=0.075 --eos-symbol=100 \
  ark:- $base/$order/train_counts.JOB || exit 1;

cat $base/$order/test_counts.* | gzip -c > $base/test_counts.gz
cat $base/$order/train_counts.* | gzip -c > $base/train_counts.gz

rm $base/$order/test_counts.*
rm $base/$order/train_counts.*

# Retrieve the top num_feats (e.g., 5000)  most common n-grams from the
# training data. The utterances will be represented as the vector of 
# soft-counts for these selected n-grams. 
local/sort_top_ngrams.pl <(gunzip -c $base/train_counts.gz) \
  | head -n $num_feats > $base/ngram_table

# Decode the training and test (LRE07) datasets as the total counts of each
# n-gram included in the ngram_table constructed above. 
# We require that the total expected count of n-grams is at least 1 for the
# training data.  The count can be small (less than 1) if the only phone 
# sequences in the utterance are very uncommon and therefore not included
# in the ngram_table. For LRE07 min_count_test=0 because we still need to
# include the utterance in our evaluation, even if we don't have a good 
# representation for it.
min_count_train=1
min_count_test=0
local/get_ngram_count.pl <(gunzip -c $base/train_counts.gz) \
  $base/ngram_table $min_count_train > $base/train_feats_unnorm.txt
local/get_ngram_count.pl <(gunzip -c $base/test_counts.gz) \
  $base/ngram_table $min_count_test > $base/test_feats_unnorm.txt

# Normalize each feature by the total frequency of that feature.
local/normalize_feats.pl $base/train_feats_unnorm.txt $base/ngram_table \
  $base/train_feats.txt
local/normalize_feats.pl $base/test_feats_unnorm.txt $base/ngram_table \
  $base/test_feats.txt

copy-vector ark,t:$base/test_feats.txt \
  ark,scp:$base/test_feats.ark,$base/test_feats.scp
copy-vector ark,t:$base/train_feats.txt \
  ark,scp:$base/train_feats.ark,$base/train_feats.scp

rm $base/test_feats*txt
rm $base/train_feats*txt

# Create new utt2lang files; some of the utterances have been lost if
# we were unable to create features for them.
utils/filter_scp.pl -f 0 $base/train_feats.scp data/train/utt2lang \
  > $base/utt2lang_train
utils/filter_scp.pl -f 0 $base/test_feats.scp data/lre07/utt2lang \
  > $base/utt2lang_test
