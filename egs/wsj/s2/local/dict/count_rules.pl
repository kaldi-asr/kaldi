#!/usr/bin/perl

# This program takes the output of score_prons.pl and collates
# it for each (rule, destress) pair so that we get the
# counts of right/partial/wrong for each pair.

# The input is a 7-tuple on each line, like:
# word;pron;base-word;base-pron;rule-name;de-stress;right|partial|wrong
#
# The output format is a 5-tuple like:
#
# rule;destress;right-count;partial-count;wrong-count
#

if (@ARGV != 0 && @ARGV != 1) {
  die "Usage: count_rules.pl < scored_candidate_prons > rule_counts";
}


while(<>) {
  chop;
  $line = $_;
  my ($word, $pron, $baseword, $basepron, $rulename, $destress, $score) = split(";", $line);
  
  my $key = $rulename . ";" . $destress;

  if (!defined $counts{$key}) {
    $counts{$key} = [ 0, 0, 0 ]; # new anonymous array.
  }
  $ref = $counts{$key};
  if ($score eq "right") {
    $$ref[0]++;
  } elsif ($score eq "partial") {
    $$ref[1]++;
  } elsif ($score eq "wrong") {
    $$ref[2]++;
  } else {
    die "Bad score $score\n";
  }
}

while ( my ($key, $value) = each(%counts)) {
  print $key . ";" . join(";", @$value) . "\n";
}
