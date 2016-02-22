#!/usr/bin/perl

# Copyright 2012  Johns Hopkins University (author: Daniel Povey)

# Generate a topology file.  This allows control of the number of states in the
# non-silence HMMs, and in the silence HMMs.

use Getopt::Long;

$nonsil_self_loop_p = 0.9;
$nonsil_transition_p = 0.1;
$sil_self_loop_p = 0.5;
$sil_transition_p = 0.5;

GetOptions('nonsil-self-loop-probability:f' => \$nonsil_self_loop_p,
           'nonsil-transition-probability:f' => \$nonsil_transition_p,
           'sil-self-loop-probability:f' => \$sil_self_loop_p,
           'sil-transition-probability:f' => \$sil_transition_p);

if(@ARGV != 4) {
  print STDERR "Usage: sid/gen_vad_topo.pl [options] <speech-duration> <silence-duration> <colon-separated-nonsilence-phones> <colon-separated-silence-phones>\n";
  print STDERR "e.g.:  sid/gen_vad_topo.pl 75 30 2 1\n";
  exit (1);
}

($num_nonsil_states, $num_sil_states, $nonsil_phones, $sil_phones) = @ARGV;

( $num_nonsil_states >= 1 && $num_nonsil_states <= 100 ) || die "Unexpected number of nonsilence-model states $num_nonsil_states\n";
( $num_sil_states >= 1 && $num_sil_states <= 100 ) || die "Unexpected number of silence-model states $num_sil_states\n";

$nonsil_phones =~ s/:/ /g;
$sil_phones =~ s/:/ /g;
$nonsil_phones =~ m/^\d[ \d]*$/ || die "$0: bad arguments @ARGV\n";
$sil_phones =~ m/^\d[ \d]*$/ || die "$0: bad arguments @ARGV\n";

print "<Topology>\n";
print "<TopologyEntry>\n";
print "<ForPhones>\n";
print "$nonsil_phones\n";
print "</ForPhones>\n";
for ($state = 0; $state < $num_nonsil_states - 1; $state++) {
  $statep1 = $state+1;
  print "<State> $state <PdfClass> 0 <Transition> $statep1 1.0 </State>\n";
}
$statep1 = $state+1;
print "<State> $state <PdfClass> 0 <Transition> $state $nonsil_self_loop_p <Transition> $statep1 $nonsil_transition_p </State>\n";
print "<State> $num_nonsil_states </State>\n"; # non-emitting final state.
print "</TopologyEntry>\n";

# Now silence phones.  
print "<TopologyEntry>\n";
print "<ForPhones>\n";
print "$sil_phones\n";
print "</ForPhones>\n";
for ($state = 0; $state < $num_sil_states - 1; $state++) {
  $statep1 = $state+1;
  print "<State> $state <PdfClass> 0 <Transition> $statep1 1.0 </State>\n";
}
$statep1 = $state+1;
print "<State> $state <PdfClass> 0 <Transition> $state $sil_self_loop_p <Transition> $statep1 $sil_transition_p </State>\n";
print "<State> $num_sil_states </State>\n"; # non-emitting final state.
print "</TopologyEntry>\n";
print "</Topology>\n";
