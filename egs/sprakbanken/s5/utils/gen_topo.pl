#!/usr/bin/perl

# Copyright 2012  Johns Hopkins University (author: Daniel Povey)

# Generate a topology file.  This allows control of the number of states in the
# non-silence HMMs, and in the silence HMMs.

if(@ARGV != 4) {
  print STDERR "Usage: utils/gen_topo.pl <num-nonsilence-states> <num-silence-states> <colon-separated-nonsilence-phones> <colon-separated-silence-phones>\n";
  print STDERR "e.g.:  utils/gen_topo.pl 3 5 4:5:6:7:8:9:10 1:2:3\n";
  exit (1);
}

($num_nonsil_states, $num_sil_states, $nonsil_phones, $sil_phones) = @ARGV;

( $num_nonsil_states >= 1 && $num_nonsil_states <= 100 ) || die "Unexpected number of nonsilence-model states $num_nonsil_states\n";
( $num_sil_states >= 3 && $num_sil_states <= 100 ) || die "Unexpected number of silence-model states $num_sil_states\n";

$nonsil_phones =~ s/:/ /g;
$sil_phones =~ s/:/ /g;
$nonsil_phones =~ m/^\d[ \d]+$/ || die "$0: bad arguments @ARGV\n";
$sil_phones =~ m/^\d[ \d]*$/ || die "$0: bad arguments @ARGV\n";

print "<Topology>\n";
print "<TopologyEntry>\n";
print "<ForPhones>\n";
print "$nonsil_phones\n";
print "</ForPhones>\n";
for ($state = 0; $state < $num_nonsil_states; $state++) {
  $statep1 = $state+1;
  print "<State> $state <PdfClass> $state <Transition> $state 0.75 <Transition> $statep1 0.25 </State>\n";
}
print "<State> $num_nonsil_states </State>\n"; # non-emitting final state.
print "</TopologyEntry>\n";
# Now silence phones.  They have a different topology-- apart from the first and
# last states, it's fully connected.
$transp = 1.0 / ($num_sil_states-1);

print "<TopologyEntry>\n";
print "<ForPhones>\n";
print "$sil_phones\n";
print "</ForPhones>\n";
print "<State> 0 <PdfClass> 0 ";
for ($nextstate = 0; $nextstate < $num_sil_states-1; $nextstate++) { # Transitions to all but last 
  # emitting state.
  print "<Transition> $nextstate $transp ";
}
print "</State>\n";
for ($state = 1; $state < $num_sil_states-1; $state++) { # the central states all have transitions to
  # themselves and to the last emitting state.
  print "<State> $state <PdfClass> $state ";
  for ($nextstate = 1; $nextstate < $num_sil_states; $nextstate++) {
    print "<Transition> $nextstate $transp ";
  }
  print "</State>\n";
}
# Final emitting state (non-skippable).
$state = $num_sil_states-1;
print "<State> $state <PdfClass> $state <Transition> $state 0.75 <Transition> $num_sil_states 0.25 </State>\n";
# Final nonemitting state:
print "<State> $num_sil_states </State>\n"; 
print "</TopologyEntry>\n";
print "</Topology>\n";
