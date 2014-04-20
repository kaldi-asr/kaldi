#!/usr/bin/perl

# This program takes candidate prons from "get_candidate_prons.pl" or
# "limit_candidate_prons.pl", and a reference dictionary covering those words,
# and outputs the same format but with scoring information added (so we go from
# 6 to 7 fields).  The scoring information says, for each generated pron,
# whether we have a match, a partial match, or no match, to some word in the
# dictionary.  A partial match means it's correct except for stress.

# The input is a 6-tuple on each line, like:
# word;pron;base-word;base-pron;rule-name;de-stress
#
# The output is the same except with one more field, the score,
# which may be "right", "wrong", "partial".

if (@ARGV != 1 && @ARGV != 2) {
  die "Usage: score_prons.pl reference_dict [candidate_prons] > scored_candidate_prons";
}

$dict = shift @ARGV;
open(D, "<$dict") || die "Opening dictionary $dict";

while(<D>) { # Set up some hashes that tell us when
  # a (word,pron) pair is correct (and the same for
  # prons with stress information removed).
  chop;
  @A = split(" ", $_);
  $word = shift @A;
  $pron = join(" ", @A);
  $pron_nostress = $pron;
  $pron_nostress =~ s:\d::g;
  $word_and_pron{$word.";".$pron} = 1;
  $word_and_pron_nostress{$word.";".$pron_nostress} = 1;
}

while(<>) {
  chop;
  $line = $_;
  my ($word, $pron, $baseword, $basepron, $rulename, $destress) = split(";", $line);
  $pron_nostress = $pron;
  $pron_nostress =~ s:\d::g;
  if (defined $word_and_pron{$word.";".$pron}) {
    $score = "right";
  } elsif (defined $word_and_pron_nostress{$word.";".$pron_nostress}) {
    $score = "partial";
  } else {
    $score = "wrong";
  }
  print $line.";".$score."\n";
}
