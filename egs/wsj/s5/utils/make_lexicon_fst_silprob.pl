#!/usr/bin/env perl
use warnings; #sed replacement for -w perl parameter
# Copyright 2010-2011  Microsoft Corporation
#                2013  Johns Hopkins University (author: Daniel Povey)
#                2015  Hainan Xu
#                2015  Guoguo Chen

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


# makes lexicon FST, in text form, from lexicon which contains (optional) 
# probabilities of pronuniations, and (mandatory) probabilities of silence 
# before and after the pronunciation. This script is almost the same with
# the make_lexicon_fst.pl script except for the word-dependent silprobs part

if (@ARGV != 4) {
  print STDERR "Usage: $0 lexiconp_disambig.txt \\\n";
  print STDERR "       silprob.txt silphone sil_disambig_sym > lexiconfst.txt \n";
  print STDERR "\n";
  print STDERR "This script is almost the same as the utils/make_lexicon_fst.pl\n";
  print STDERR "except here we include word-dependent silence probabilities\n";
  print STDERR "when making the lexicon FSTs.\n";
  exit(1);
}

$lexfn = shift @ARGV;
$silprobfile = shift @ARGV;

($silphone,$sildisambig) = @ARGV;

open(L, "<$lexfn") || die "Error opening lexicon $lexfn";
open(SP, "<$silprobfile") || die "Error opening word-sil-probs $SP";

sub is_sil {
  # Return true (1) if provided with a phone-sequence
  # that means silence.
  # @_ is the parameters of the function
  # This function returns true if @_ equals ( $silphone )
  # or something of the form ( "#0", $silphone, "#1" )
  # where the "#0" and "#1" are disambiguation symbols.
  return ( @_ == 1 && $_[0] eq $silphone ||
           (@_ == 3 && $_[1] eq $silphone &&
            $_[0] =~ m/^\#\d+$/ &&
            $_[0] =~ m/^\#\d+$/));
}

$silbeginprob = -1;
$silendcorrection = -1;
$nonsilendcorrection = -1;
$siloverallprob = -1;

while (<SP>) {
  @A = split(" ", $_);
  $w = shift @A;
  if ($w eq "<s>") {
    $silbeginprob = shift @A;
  } 
  if ($w eq "</s>_s") {
    $silendcorrection = shift @A;
  }
  if ($w eq "</s>_n") {
    $nonsilendcorrection = shift @A;
  }
  if ($w eq "overall") {
    $siloverallprob = shift @A;
  }
}

$startstate = 0;
$nonsilstart = 1;
$silstart = 2;
$nextstate = 3;

$cost = -log($silbeginprob);
print "$startstate\t$silstart\t$silphone\t<eps>\t$cost\n"; # will change these
$cost = -log(1 - $silbeginprob);
print "$startstate\t$nonsilstart\t$sildisambig\t<eps>\t$cost\n";

while (<L>) {
  @A = split(" ", $_);
  $w = shift @A;
  $pron_prob = shift @A;
  if (! defined $pron_prob || !($pron_prob > 0.0 && $pron_prob <= 1.0)) {
    die "Bad pronunciation probability in line $_";
  }

  $wordsilprob = shift @A;
  $silwordcorrection = shift @A;
  $nonsilwordcorrection = shift @A;

  $pron_cost = -log($pron_prob);
  $wordsilcost = -log($wordsilprob);
  $wordnonsilcost = -log(1.0 - $wordsilprob);
  $silwordcost = -log($silwordcorrection);
  $nonsilwordcost = -log($nonsilwordcorrection);

  $first = 1;  # used as a bool, to handle the first phone (adding sils)
  while (@A > 0) {
    $p = shift @A;

    if ($first == 1) {
      $newstate = $nextstate++;

      # for nonsil before w
      $cost = $nonsilwordcost + $pron_cost;
      print "$nonsilstart\t$newstate\t$p\t$w\t$cost\n";

      # for sil before w
      $cost = $silwordcost + $pron_cost;
      print "$silstart\t$newstate\t$p\t$w\t$cost\n";
      $first = 0;
    }
    else {
      $oldstate = $nextstate - 1;
      print "$oldstate\t$nextstate\t$p\t<eps>\n";
      $nextstate++;
    }
    if (@A == 0) {
      $oldstate = $nextstate - 1;
      # for no sil after w
      $cost = $wordnonsilcost;
      print "$oldstate\t$nonsilstart\t$sildisambig\t<eps>\t$cost\n";

      # for sil after w
      $cost = $wordsilcost;
      print "$oldstate\t$silstart\t$silphone\t<eps>\t$cost\n";
    }
  }
}
$cost = -log($silendcorrection);
print "$silstart\t$cost\n";   
$cost = -log($nonsilendcorrection);
print "$nonsilstart\t$cost\n";
