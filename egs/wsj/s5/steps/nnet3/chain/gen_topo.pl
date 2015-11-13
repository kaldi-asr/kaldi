#!/usr/bin/env perl

# Copyright 2012  Johns Hopkins University (author: Daniel Povey)

# Generate a topology file.  This allows control of the number of states in the
# non-silence HMMs, and in the silence HMMs.  This is a modified version of
# 'utils/gen_topo.pl' that generates a different type of topology, one that we
# believe should be useful in the 'chain' model.  Note: right now it doesn't
# have any real options, and it treats silence and nonsilence the same.  The
# intention is that you write different versions of this script, or add options,
# if you experiment with it.

if (@ARGV != 2) {
  print STDERR "Usage: utils/gen_topo.pl <colon-separated-nonsilence-phones> <colon-separated-silence-phones>\n";
  print STDERR "e.g.:  utils/gen_topo.pl 4:5:6:7:8:9:10 1:2:3\n";
  exit (1);
}

($nonsil_phones, $sil_phones) = @ARGV;

$nonsil_phones =~ s/:/ /g;
$sil_phones =~ s/:/ /g;
$nonsil_phones =~ m/^\d[ \d]+$/ || die "$0: bad arguments @ARGV\n";
$sil_phones =~ m/^\d[ \d]*$/ || die "$0: bad arguments @ARGV\n";

print "<Topology>\n";
print "<TopologyEntry>\n";
print "<ForPhones>\n";
print "$nonsil_phones $sil_phones\n";
print "</ForPhones>\n";
# The next two lines may look like a bug, but they are as intended.  State 0 has
# no self-loop, it happens exactly once.  And it can go either to state 1 (with
# a self-loop) or to state 2, so we can have zero or more instances of state 1
# following state 0.
# We make the transition-probs 0.5 so they normalize, to keep the code happy.
# In fact, we always set the transition probability scale to 0.0 in the 'chain'
# code, so they are never used.
print "<State> 0 <PdfClass> 0 <Transition> 1 0.5 <Transition> 2 0.5 </State>\n";
print "<State> 1 <PdfClass> 1 <Transition> 1 0.5 <Transition> 2 0.5 </State>\n";
print "<State> 2 </State>\n";
print "</TopologyEntry>\n";
print "</Topology>\n";
