#!/usr/bin/env perl

# This program takes the output of count_rules.pl, which is tuples
# of the form
#
# rule;destress;right-count;partial-count;wrong-count
#
# and outputs lines of the form
#
# rule;de-stress;score
#
# where the score, between 0 and 1 (1 better), is 
# equal to:
#
# It forms a score between 0 and 1, of the form:
# ((#correct) +  $partial_score * (#partial)) / (#correct + #partial + #wrong + $ballast)
#
# where $partial_score (e.g. 0.8) is the score we assign to a "partial" match,
# and $ballast is a small number, e.g. 1, that is treated like "extra" wrong scores, to penalize
# rules with few observations.
#
# It outputs all rules that at are at least the

$ballast = 1;
$partial_score = 0.8;
$destress_penalty = 1.0e-05; # Give destressed rules a small
# penalty vs. their no-destress counterparts, so if we
# have to choose arbitrarily we won't destress (seems safer)>

for ($n = 1; $n <= 4; $n++) {
  if ($ARGV[0] eq "--ballast") {
    shift @ARGV;
    $ballast = shift @ARGV;
  }
  if ($ARGV[0] eq "--partial-score") {
    shift @ARGV;
    $partial_score = shift @ARGV;
    ($partial_score >= 0.0 && $partial_score <= 1.0) || die "Invalid partial_score: $partial_score";
  }
}

(@ARGV == 0 || @ARGV == 1) || die "Usage: score_rules.pl [--ballast ballast-count] [--partial-score partial-score] [input from count_rules.pl]";

while(<>) {
  @A = split(";", $_);
  @A == 5 || die "Bad input line; $_";
  ($rule,$destress,$right_count,$partial_count,$wrong_count) = @A;
  $rule_score = ($right_count + $partial_score*$partial_count) / 
    ($right_count+$partial_count+$wrong_count+$ballast);
  if ($destress eq "yes") { $rule_score -= $destress_penalty; }
  print join(";", $rule, $destress, sprintf("%.5f", $rule_score)) . "\n";
}
