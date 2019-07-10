#!/usr/bin/env perl

# Copyright 2012  Johns Hopkins University (author: Daniel Povey)

# Generate a topology file.  This allows control of the number of states in the
# non-silence HMMs, and in the silence HMMs.
# This is the topology we use for GMM training, which is, when configured
# with 3 states, the Bakis model.  For chain (lattice-free MMI) training, see
# steps/chain/gen_topo.pl.

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
$nonsil_phones =~ m/^\d[ \d]+$/ || die "$0: bad arguments @ARGV\n";
$sil_phones =~ m/^\d[ \d]*$/ || die "$0: bad arguments @ARGV\n";

print "<Topology>\n";
print "<TopologyEntry>\n";
print "<ForPhones>\n";
print "$nonsil_phones\n";
print "</ForPhones>\n";
# The following is the single transition leaving the start-state.  It has pdf-id
# 1, corresponding to state 1 which it enters..  The cost is 0.0 = log(1); there
# is only one choice here.  Note: there are actually $num_nonsil_states + 1
# states, but in HMM terms it's equivalent to $num_nonsil_states states;
# and that's the length of the shortest successful path.
print "0  1  1  0.0\n";
for ($state = 1; $state <= $num_nonsil_states; $state++) {
  $pdf_class = $state;
  $next_state = $state + 1;
  $next_pdf_class = $next_state;
  # self-loop.
  print "$state $state $pdf_class 0.6931471806\n";
  if ($next_state <= $num_nonsil_states) {
    print "$state $next_state $next_pdf_class 0.6931471806\n";
  } else {
    print "$state 0.6931471806\n";  # final-prob.
  }
}
print "\n";  # terminate the FSA.. empty line marks its end.
print "</TopologyEntry>\n";
# Now silence phones.  They have a different topology-- apart from the first and
# last states, it's fully connected, as long as you have >= 3 states.

print "<TopologyEntry>\n";
print "<ForPhones>\n";
print "$sil_phones\n";
print "</ForPhones>\n";


print "0  1  1  0.0\n";
if ($num_sil_states > 1) {
  # Note: actually it must be >= 3, we checked this above;
  # 2 is disallowed (I know, it's odd).
  # Also note: $num_sil_states is not actually the number of states
  # in the FSA; it's the number of states in its HMM equivalent.
  # the FSA has one extra state, state 0.
  # we'll treat the final state, numbered $num_sil_states,
  # separately; it doesn't have the transition back to
  # lower-numbered states.

  $self_loop_cost = 0.6931471806;  # -log(0.5)
  $non_self_loop_cost = -log(0.5 / ($num_sil_states - 2));

  $state = 1;
  $pdf_id = $state;
  print "$state  $state  $pdf_id  $self_loop_cost\n";
  for ($next_state = 2; $next_state < $num_sil_states; $next_state++) {
    $next_pdf_id = $next_state;
    print "$state  $next_state  $next_pdf_id  $non_self_loop_cost\n";
  }

  for ($state = 2; $state < $num_sil_states; $state++) {
    $pdf_id = $state;
    for ($next_state = 2; $next_state <= $num_sil_states; $next_state++) {
      my $cost = ($next_state == $state ? $self_loop_cost : $non_self_loop_cost);
      $next_pdf_id = $next_state;
      print "$state  $next_state  $next_pdf_id  $cost\n";
    }
  }
  $final_state = $num_sil_states;
  $pdf_id = $final_state;
  print "$final_state  $final_state  $pdf_id  $self_loop_cost\n";
  print "$final_state 0.6931471806\n";
  print "\n";
} else {
  print "0  0  1  0.6931471806\n";
  print "1  1  1  0.6931471806\n";
  print "1  0.6931471806\n";
  print "\n";
}
print "</TopologyEntry>\n";
print "</Topology>\n";
