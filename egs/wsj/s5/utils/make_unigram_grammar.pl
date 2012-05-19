#!/usr/bin/perl
# Copyright 2010-2012 Microsoft Corporation  Johns Hopkins University (Author: Daniel Povey)

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

# This script is used in discriminative training.
# This script makes a simple unigram-loop version of G.fst
# using a unigram grammar estimated from some training transcripts.
# This is for MMI training.
# We don't have any silences in G.fst; these are supplied by the
# optional silences in the lexicon.

# Note: the symbols in the transcripts become the input and output
# symbols of G.txt; these can be numeric or not.

if(@ARGV != 0) {
    die "Usage: make_unigram_grammar.pl < text-transcripts > G.txt"
}

$totcount = 0;
$nl = 0;
while (<>) {
  @A = split(" ", $_);
  foreach $a (@A) {
    $count{$a}++;
    $totcount++;
  }
  $nl++;
  $totcount++; # Treat end-of-sentence as a symbol for purposes of
  # $totcount, so the grammar is properly stochastic.  This doesn't
  # become </s>, it just becomes the final-prob.
}

foreach $a (keys %count) {
  $prob = $count{$a} / $totcount;
  $cost = -log($prob);          # Negated natural-log probs.
  print "0\t0\t$a\t$a\t$cost\n";
}
# Zero final-cost.
$final_prob = $nl / $totcount;
$final_cost = -log($final_prob);
print "0\t$final_cost\n";

