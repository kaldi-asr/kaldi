#!/usr/bin/env perl
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


# Adds disambiguation symbols to a lexicon.
# Outputs still in the normal lexicon format.
# Disambig syms are numbered #1, #2, #3, etc. (#0 
# reserved for symbol in grammar).
# Outputs the number of disambig syms to the standard output.
# With the --pron-probs option, expects the second field
# of each lexicon line to be a pron-prob.

$pron_probs = 0;
$sil_probs = 0;

if ($ARGV[0] eq "--pron-probs") {
  $pron_probs = 1;
  shift @ARGV;
}

if ($ARGV[0] eq "--sil-probs") {
  $sil_probs = 1;
  shift @ARGV;
}

if(@ARGV != 2) {
    die "Usage: add_lex_disambig.pl [--pron-probs] [--sil-probs] lexicon.txt lexicon_disambig.txt "
}


$lexfn = shift @ARGV;
$lexoutfn = shift @ARGV;

open(L, "<$lexfn") || die "Error opening lexicon $lexfn";

# (1)  Read in the lexicon.
@L = ( );
while(<L>) {
    @A = split(" ", $_);
    push @L, join(" ", @A);
}

# (2) Work out the count of each phone-sequence in the
# lexicon.

foreach $l (@L) {
    @A = split(" ", $l);
    shift @A; # Remove word.
    if ($pron_probs) {
      $p = shift @A;
      if (!($p > 0.0 && $p <= 1.0)) { die "Bad lexicon line $l (expecting pron-prob as second field)"; }
    }
    if ($sil_probs) {
      $silp = shift @A;
      if (!($silp > 0.0 && $silp <= 1.0)) { die "Bad lexicon line $l for silprobs"; }
      $correction = shift @A;
      if ($correction <= 0.0) { die "Bad lexicon line $l for silprobs"; }
      $correction = shift @A;
      if ($correction <= 0.0) { die "Bad lexicon line $l for silprobs"; }
    }
    if (!(@A)) {
      die "Bad lexicon line $1, no phone in phone list";
    }
    $count{join(" ",@A)}++;
}

# (3) For each left sub-sequence of each phone-sequence, note down
# that exists (for identifying prefixes of longer strings).

foreach $l (@L) {
    @A = split(" ", $l);
    shift @A; # Remove word.
    if ($pron_probs) { shift @A; } # remove pron-prob.
    if ($sil_probs) {
      shift @A; # Remove silprob
      shift @A; # Remove silprob
    }
    while(@A > 0) {
        pop @A;  # Remove last phone
        $issubseq{join(" ",@A)} = 1;
    }
}

# (4) For each entry in the lexicon:
#  if the phone sequence is unique and is not a
#  prefix of another word, no diambig symbol.
#  Else output #1, or #2, #3, ... if the same phone-seq
#  has already been assigned a disambig symbol.


open(O, ">$lexoutfn") || die "Opening lexicon file $lexoutfn for writing.\n";

$max_disambig = 0;
foreach $l (@L) {
    @A = split(" ", $l);
    $word = shift @A;
    if ($pron_probs) { $pron_prob = shift @A; }
    if ($sil_probs) {
      $sil_word_prob = shift @A;
      $word_sil_correction = shift @A;
      $prev_nonsil_correction = shift @A
    }
    $phnseq = join(" ",@A);
    if(!defined $issubseq{$phnseq}
       && $count{$phnseq} == 1) {
        ; # Do nothing.
    } else {
        if($phnseq eq "") { # need disambig symbols for the empty string
            # that are not use anywhere else.
            $max_disambig++;
            $reserved{$max_disambig} = 1;
            $phnseq = "#$max_disambig";
        } else {
            $curnumber = $disambig_of{$phnseq};
            if(!defined{$curnumber}) { $curnumber = 0; }
            $curnumber++; # now 1 or 2, ... 
            while(defined $reserved{$curnumber} ) { $curnumber++; } # skip over reserved symbols
            if($curnumber > $max_disambig) {
                $max_disambig = $curnumber;
            }
            $disambig_of{$phnseq} = $curnumber;
            $phnseq = $phnseq . " #" . $curnumber;
         }
    }
    if ($pron_probs) {  
      if ($sil_probs) {
        print O "$word\t$pron_prob\t$sil_word_prob\t$word_sil_correction\t$prev_nonsil_correction\t$phnseq\n"; 
      }
      else {
        print O "$word\t$pron_prob\t$phnseq\n"; 
      }
    }
    else {  print O "$word\t$phnseq\n"; }
}

print $max_disambig . "\n";

