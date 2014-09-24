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
# This script parses the raw counts from the binary lattice-to-ngram-counts. It
# accumulates the total counts for each unique n-gram and prints them sorted in
# order of most to least common n-grams.

$train = $ARGV[0];
%phone_ngrams = ();
@phone_list = {};
# Ignore the digit phones (see ../data/lang/phones.txt).
%ignore = ("44"=>1,"45"=>1,"46"=>1
           ,"47"=>1,"48"=>1,"49"=>1,"50"=>1,"51"=>1,"52"=>1,"53"=>1,"100"=>1);
# Don't allow sil, laughter or noise to be center phones.
%noncenter = ("1" => 1, "2" => 1, "3" => 1);

open(TRAIN, "<", $train) or die "cannot open training data";
while ($decoding = <TRAIN>) {
  chomp($decoding);
  @A = split(" ", $decoding); 
  # Remove utterance id
  splice @A, 0, 1;
  for ($i = 0; $i < @A; $i++) {
    ($ngram, $prob) = split(":", $A[$i]);
    $skip = 0;
    @phones = split(",", $ngram);
    if (@phones == 1) {
      $skip = 1;
    }
    for ($j = 0; $j < @phones; $j++) {
      $phone = $phones[$j];
      chomp($phone);
      if (exists $ignore{"$phone"} ) {
        $skip = 1;
      }
      if (($j > 0) && ($j < @phones-1)
        && (exists $noncenter{"$phone"})) {
        $skip = 1;
      }
    }
    
    if ($skip == 1) {
      next;
    }

    if (exists $phone_ngrams{$ngram}) {
      $phone_ngrams{$ngram} += $prob;
    } else {
      $phone_ngrams{$ngram} = 0.0 + $prob;
    }
  }
}
close(TRAIN) || die;

@ngrams_sorted = reverse sort { $phone_ngrams{$a} 
  <=> $phone_ngrams{$b} } keys %phone_ngrams;

foreach $ngram (@ngrams_sorted) {
  print "$ngram $phone_ngrams{$ngram}\n";
}
