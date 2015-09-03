#!/usr/bin/env perl

# This takes the output of e.g. get_candidate_prons.pl or limit_candidate_prons.pl
# or reverse_candidates.pl, which is 7-tuples, one per line, of the form:
#
# word;pron;base-word;base-pron;rule-name;de-stress;rule-score
#
# and selects the most likely prons for the words based on rule
# score.  It outputs in the same format as the input (thus, it is
# similar to limit_candidates.pl in its input and output format,
# except it has a different way of selecting the prons to put out).
#
# This script will select the $max_prons best pronunciations for
# each candidate word, subject to the constraint that no pron should
# have a rule score worse than $min_rule_score.
# It first merges the candidates by, if there are multiple candidates
# generating the same pron, selecting the candidate that had the
# best associated score.  It then sorts the prons on score and
# selects the n best prons (but doesn't print out candidates with
# score beneath the threshold).


$max_prons = 4;
$min_rule_score = 0.35;


for ($n = 1; $n <= 3; $n++) {
  if ($ARGV[0] eq "--max-prons") {
    shift @ARGV;
    $max_prons = shift @ARGV;
  }
  if ($ARGV[0] eq "--min-rule-score") {
    shift @ARGV;
    $min_rule_score = shift @ARGV;
  }
}

if (@ARGV != 0 && @ARGV != 1) {
  die "Usage: select_candidates_prons.pl [candidate_prons] > selected_candidate_prons";
}

sub process_word;

undef $cur_word;
@cur_lines = ();

while(<>) {
  # input, output is:
  # word;pron;base-word;base-pron;rule-name;destress;score
  chop;
  m:^([^;]+);: || die "Unexpected input: $_";
  $word = $1;
  if (!defined $cur_word || $word eq $cur_word) {
    if (!defined $cur_word) { $cur_word = $word; }
    push @cur_lines, $_;
  } else {
    process_word(@cur_lines); # Process a series of suggested prons
    # for a particular word.
    $cur_word = $word;
    @cur_lines = ( $_ ); 
  }
}
process_word(@cur_lines);


sub process_word {
  my %pron2rule_score; # hash from generated pron to rule score for that pron.
  my %pron2line; # hash from generated pron to best line for that pron.
  my @cur_lines = @_;
  foreach my $line (@cur_lines) {
    my ($word, $pron, $baseword, $basepron, $rulename, $destress, $rule_score) = split(";", $line);
    if (!defined $pron2rule_score{$pron} ||
        $rule_score > $pron2rule_score{$pron}) {
      $pron2rule_score{$pron} = $rule_score;
      $pron2line{$pron} = $line;
    }
  }
  my @prons = sort { $pron2rule_score{$b} <=> $pron2rule_score{$a} } keys %pron2rule_score;
  for (my $n = 0; $n < @prons && $n < $max_prons &&
       $pron2rule_score{$prons[$n]} >= $min_rule_score; $n++) {
    print $pron2line{$prons[$n]} . "\n";
  }
}

