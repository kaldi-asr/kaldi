#!/usr/bin/perl
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
#
# This script takes the raw n-gram counts from lattice-to-ngram-counts and a
# list of n-grams we want to use as features, and constructs vectors for each
# utterance which are the expected count of each n-gram in the list.
# For example, the list of n-grams could be the top 10,000 most common trigrams
# found in the training data. Then, the feature vectors are the expected counts 
# of those 10,000 trigrams in the input.  The threshold is the minimum total 
# n-grams we will tolerate for an utterance.  For example, if threshold=1.0 and
# the total count of all the n-grams listed by ngram_feats, we don't emit a 
# feature for this entry.

$data = $ARGV[0];
$ngram_feats = $ARGV[1]; # These are the ngrams we'll use to
                          # generate the features. Could be top 1000 most
                          # common ngrams in the training data.
$threshold = $ARGV[2];

%ngrams = ();
@phone_list = {};
open(NGRAM_FEATS, "<", $ngram_feats) or die "cannot open phone table";
while (<NGRAM_FEATS>) {
  chomp($_);
  @A = split(" ", $_);
  $key = $A[0];
  $ngrams{$key} = 0.0;
}
close(NGRAM_FEATS) or die;

open(DATA, "<", $data) or die "cannot open data";
while (<DATA>) {
  chomp($_);
  @A = split(" ", $_);
  # Remove utterance id
  $uttid = splice @A, 0, 1;
  %ngram_count = %ngrams; # ngrams is initialized to 0
  $total = 0.0;
  for ($i = 1; $i < @A; $i++) {
    ($ngram, $prob) = split(":", $A[$i]);
    if (exists $ngrams{$ngram}) {
      $ngram_count{$ngram} += $prob;
      $total += $prob;
    }
  }
  if ($total < $threshold) {
    warn "Skipping utterance $utt due to a total of $total ngrams (less than $threshold)";
    next;
  }
  print "$uttid  [";
  for $ngram (keys %ngram_count) {
    if ($total < 1.0**-999) {
      $freq = 0.0;
    } else {
      $freq = $ngram_count{$ngram} / sqrt($total);
    }
    #Should probably be printing to file instead
    print " $freq";
  }
  print " ]\n";
}
close(DATA) or die;
  
