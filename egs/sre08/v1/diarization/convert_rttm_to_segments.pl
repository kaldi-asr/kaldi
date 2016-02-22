#!/usr/bin/perl -w
# Copyright 2015  Vimal Manohar (Johns Hopkins University)
# Apache 2.0.

use strict;
use POSIX;
use Getopt::Long;
use File::Basename;
use Pod::Usage;

my $help = 0;

my $frame_shift = 0.01;
my $ignore_boundaries = "false";

GetOptions('frame-shift:f' => \$frame_shift,
           'ignore-boundaries:s' => \$ignore_boundaries,
            'help|?' => \$help);

if ((@ARGV > 1 || $help)) {
  print STDERR "$0:\n" .
               "Usage: convert_rttm_to_segments.pl [options] [rttm] > <segments-out>\n";
  exit 0 if $help;
  exit 1;
} 

($frame_shift > 0.0001 && $frame_shift <= 1.0) ||
  die "Very strange frame-shift value '$frame_shift'";
($ignore_boundaries eq "false" || $ignore_boundaries eq "true") || 
  die "ignore-boundaries must be (true|false)";

while (<>) {
  chomp;
  my @A = split;
  my $file = $A[1];

  if ($A[0] =~ m/SPKR-INFO/) {
    print STDERR "Reading RTTM for file $file\n";
    next;
  } elsif ($A[0] !~ m/SPEAKER/) {
    next;
  }
  my $start_frame = floor($A[3] / $frame_shift);
  my $end_frame = floor(($A[3] + $A[4]) / $frame_shift);

  my $utt_id = sprintf("$file-%06d-%06d", $start_frame, $end_frame);

  printf STDOUT ("$utt_id $file %6.3f %6.3f\n", $A[3], $A[3] + $A[4]) or die;
}
