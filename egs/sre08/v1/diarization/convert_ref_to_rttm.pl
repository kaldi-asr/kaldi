#!/usr/bin/perl -w
# Copyright 2015  Vimal Manohar (Johns Hopkins University)
# Apache 2.0.

use strict;
use Getopt::Long;
use File::Basename;

if (@ARGV != 1) {
  print STDERR "$0:\n" .
               "Usage: convert_ref_to_rttm.pl [options] <file-name> > <rttm-out>\n";
  exit 1;
}

my $filename = $ARGV[0];

print STDERR "Extracting RTTM from ref $filename\n";

my $basename = basename($filename);
(my $utt_id = $basename) =~ s/\.[^.]+$//;

open IN, $filename or die "Could not open $filename";

my %seen_spkrs = ();

while (<IN>) {
  chomp;
  my @A = split;
  if (! defined $seen_spkrs{$A[2]}) {
    printf STDOUT ("SPKR-INFO $utt_id 1 <NA> <NA> <NA> unknown $A[2] <NA>\n");
    $seen_spkrs{$A[2]} = 1;
  }
  
  printf STDOUT ("SPEAKER $utt_id 1 %5.2f %5.2f <NA> <NA> $A[2] <NA>\n", $A[0], $A[1] - $A[0]);
}
