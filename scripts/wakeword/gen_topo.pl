#!/usr/bin/env perl

# Copyright 2012-2020  Daniel Povey
#           2018-2020  Yiming Wang

# This script is modified from utils/gen_topo.pl. It is used for wake word detection
# chain training, where the transotion probs are even. The topo also contains separate
# ForwardPdfClass and SelfLoopPdfClass.

if (@ARGV != 4) {
  print STDERR "Usage: utils/gen_topo.pl <num-nonsilence-states> <num-silence-states> <colon-separated-nonsilence-phones> <colon-separated-silence-phones>\n";
  print STDERR "e.g.:  utils/gen_topo.pl 3 5 4:5:6:7:8:9:10 1:2:3\n";
  exit (1);
}

($num_nonsil_states, $num_sil_states, $nonsil_phones, $sil_phones) = @ARGV;

( $num_nonsil_states >= 1 && $num_nonsil_states <= 100 ) ||
  die "Unexpected number of nonsilence-model states $num_nonsil_states\n";
(( $num_sil_states == 1 || $num_sil_states >= 3) && $num_sil_states <= 100 ) ||
  die "Unexpected number of silence-model states $num_sil_states\n";

$nonsil_phones =~ s/:/ /g;
$sil_phones =~ s/:/ /g;
$nonsil_phones =~ m/^\d[ \d]*$/ || die "$0: bad arguments @ARGV\n";
$sil_phones =~ m/^\d[ \d]*$/ || die "$0: bad arguments @ARGV\n";

print "<Topology>\n";
print "<TopologyEntry>\n";
print "<ForPhones>\n";
print "$nonsil_phones\n";
print "</ForPhones>\n";
for ($state = 0; $state < $num_nonsil_states; $state++) {
  $statep1 = $state+1;
  print "<State> $state <ForwardPdfClass> @{[2*$state]} <SelfLoopPdfClass> @{[2*$state+1]} <Transition> $state 0.5 <Transition> $statep1 0.5 </State>\n";
}
print "<State> $num_nonsil_states </State>\n"; # non-emitting final state.
print "</TopologyEntry>\n";
# Now silence phones.  They have a different topology-- apart from the first and
# last states, it's fully connected, as long as you have >= 3 states.

if ($num_sil_states > 1) {
  $transp = 1.0 / ($num_sil_states-1);
  print "<TopologyEntry>\n";
  print "<ForPhones>\n";
  print "$sil_phones\n";
  print "</ForPhones>\n";
  print "<State> 0 <PdfClass> <ForwardPdfClass> 0 <SelfLoopPdfClass> 1 ";
  for ($nextstate = 0; $nextstate < $num_sil_states-1; $nextstate++) { # Transitions to all but last
    # emitting state.
    print "<Transition> $nextstate $transp ";
  }
  print "</State>\n";
  for ($state = 1; $state < $num_sil_states-1; $state++) { # the central states all have transitions to
    # themselves and to the last emitting state.
    print "<State> $state <ForwardPdfClass> @{[2*$state]} <SelfLoopPdfClass> @{[2*$state+1]} ";
    for ($nextstate = 1; $nextstate < $num_sil_states; $nextstate++) {
      print "<Transition> $nextstate $transp ";
    }
    print "</State>\n";
  }
  # Final emitting state (non-skippable).
  $state = $num_sil_states-1;
  print "<State> $state <ForwardPdfClass> @{[2*$state]} <SelfLoopPdfClass> @{[2*$state+1]} <Transition> $state 0.5 <Transition> $num_sil_states 0.5 </State>\n";
  # Final nonemitting state:
  print "<State> $num_sil_states </State>\n";
  print "</TopologyEntry>\n";
} else {
  print "<TopologyEntry>\n";
  print "<ForPhones>\n";
  print "$sil_phones\n";
  print "</ForPhones>\n";
  print "<State> 0 <ForwardPdfClass> 0 <SelfLoopPdfClass> 1 ";
  print "<Transition> 0 0.5 ";
  print "<Transition> 1 0.5 ";
  print "</State>\n";
  print "<State> $num_sil_states </State>\n"; # non-emitting final state.
  print "</TopologyEntry>\n";
}

print "</Topology>\n";
