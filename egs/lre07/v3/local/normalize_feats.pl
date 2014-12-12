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
# This script normalizes the n-gram features by dividing by the square root of
# the frequency of the n-gram in the training data. 

$data = $ARGV[0];
$ngram_table = $ARGV[1];
$out = $ARGV[2];

%counts = ();
$tot = 0.0;

open(STATS, "<", $ngram_table) or die "cannot open $ngram_table";
$feat = 0;
while(<STATS>) {
  chomp($_);
  ($ngram, $count) = split(" ", $_);
  $count{$feat} = $count;
  $tot += $count;
  $feat += 1;
}

close(STATS) or die;

open(DATA, "<", $data) or die "cannot open $data";
open(OUT, ">", $out) or die "cannot open $out";
while (<DATA>) {
  chomp($_);
  @A = split(" ", $_);
  $uttid = $A[0];
  @A= @A[2 .. $#A-1];
  print OUT "$uttid [ ";
  for ($i = 0; $i < @A; $i++) {
    # Divide by the sqrt of the frequency of that
    # ngram.
    $normalized = $A[$i]/sqrt($count{$i}/$tot);
    print OUT "$normalized ";
  }
  print OUT "]\n";
}
close(DATA) or die;
close(OUT) or die;
