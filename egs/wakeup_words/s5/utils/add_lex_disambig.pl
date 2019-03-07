#!/usr/bin/env perl
#  Copyright 2010-2011  Microsoft Corporation
#            2013-2016  Johns Hopkins University (author: Daniel Povey)
#                 2015  Hainan Xu
#                 2015  Guoguo Chen

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
# With the --sil-probs option, expects three additional
# fields after the pron-prob, representing various components
# of the silence probability model.

$pron_probs = 0;
$sil_probs = 0;
$first_allowed_disambig = 1;

for ($n = 1; $n <= 3 && @ARGV > 0; $n++) {
  if ($ARGV[0] eq "--pron-probs") {
    $pron_probs = 1;
    shift @ARGV;
  }
  if ($ARGV[0] eq "--sil-probs") {
    $sil_probs = 1;
    shift @ARGV;
  }
  if ($ARGV[0] eq "--first-allowed-disambig") {
    $first_allowed_disambig = 0 + $ARGV[1];
    if ($first_allowed_disambig < 1) {
      die "add_lex_disambig.pl: invalid --first-allowed-disambig option: $first_allowed_disambig\n";
    }
    shift @ARGV;
    shift @ARGV;
  }
}

if (@ARGV != 2) {
  die "Usage: add_lex_disambig.pl [opts] <lexicon-in> <lexicon-out>\n" .
    "This script adds disambiguation symbols to a lexicon in order to\n" .
    "make decoding graphs determinizable; it adds pseudo-phone\n" .
    "disambiguation symbols #1, #2 and so on at the ends of phones\n" .
    "to ensure that all pronunciations are different, and that none\n" .
    "is a prefix of another.\n" .
    "It prints to the standard output the number of the largest-numbered" .
    "disambiguation symbol that was used.\n" .
    "\n" .
    "Options:   --pron-probs       Expect pronunciation probabilities in the 2nd field\n" .
    "           --sil-probs        [should be with --pron-probs option]\n" .
    "                              Expect 3 extra fields after the pron-probs, for aspects of\n" .
    "                              the silence probability model\n" .
    "           --first-allowed-disambig <n>  The number of the first disambiguation symbol\n" .
    "                              that this script is allowed to add.  By default this is\n" .
    "                              #1, but you can set this to a larger value using this option.\n" .
    "e.g.:\n" .
    " add_lex_disambig.pl lexicon.txt lexicon_disambig.txt\n" .
    " add_lex_disambig.pl --pron-probs lexiconp.txt lexiconp_disambig.txt\n" .
    " add_lex_disambig.pl --pron-probs --sil-probs lexiconp_silprob.txt lexiconp_silprob_disambig.txt\n";
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
# that it exists (for identifying prefixes of longer strings).

foreach $l (@L) {
    @A = split(" ", $l);
    shift @A; # Remove word.
    if ($pron_probs) { shift @A; } # remove pron-prob.
    if ($sil_probs) {
      shift @A; # Remove silprob
      shift @A; # Remove silprob
      shift @A; # Remove silprob, there three numbers for sil_probs
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

# max_disambig will always be the highest-numbered disambiguation symbol that
# has been used so far.
$max_disambig = $first_allowed_disambig - 1;

foreach $l (@L) {
  @A = split(" ", $l);
  $word = shift @A;
  if ($pron_probs) {
    $pron_prob = shift @A;
  }
  if ($sil_probs) {
    $sil_word_prob = shift @A;
    $word_sil_correction = shift @A;
    $prev_nonsil_correction = shift @A
  }
  $phnseq = join(" ", @A);
  if (!defined $issubseq{$phnseq}
      && $count{$phnseq} == 1) {
    ;                           # Do nothing.
  } else {
    if ($phnseq eq "") {        # need disambig symbols for the empty string
      # that are not use anywhere else.
      $max_disambig++;
      $reserved_for_the_empty_string{$max_disambig} = 1;
      $phnseq = "#$max_disambig";
    } else {
      $cur_disambig = $last_used_disambig_symbol_of{$phnseq};
      if (!defined $cur_disambig) {
        $cur_disambig = $first_allowed_disambig;
      } else {
        $cur_disambig++;           # Get a number that has not been used yet for
                                   # this phone sequence.
      }
      while (defined $reserved_for_the_empty_string{$cur_disambig}) {
        $cur_disambig++;
      }
      if ($cur_disambig > $max_disambig) {
        $max_disambig = $cur_disambig;
      }
      $last_used_disambig_symbol_of{$phnseq} = $cur_disambig;
      $phnseq = $phnseq . " #" . $cur_disambig;
    }
  }
  if ($pron_probs) {
    if ($sil_probs) {
      print O "$word\t$pron_prob\t$sil_word_prob\t$word_sil_correction\t$prev_nonsil_correction\t$phnseq\n";
    } else {
      print O "$word\t$pron_prob\t$phnseq\n";
    }
  } else {
    print O "$word\t$phnseq\n";
  }
}

print $max_disambig . "\n";
