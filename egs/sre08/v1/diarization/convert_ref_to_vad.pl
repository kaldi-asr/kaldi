#!/usr/bin/perl -w
# Copyright 2015  Vimal Manohar (Johns Hopkins University)
# Apache 2.0.

use strict;
use Getopt::Long;
use File::Basename;

my $frame_shift = 0.01;

GetOptions('frame-shift:f' => \$frame_shift);

if (@ARGV != 1) {
  print STDERR "$0:\n" .
               "Usage: convert_ref_to_vad.pl [options] <file-name> > <vad-out>\n";
  exit 1;
}

($frame_shift > 0.0001 && $frame_shift <= 1.0) ||
  die "Very strange frame-shift value '$frame_shift'";

my $filename = $ARGV[0];

print STDERR "Extracting VAD from ref $filename\n";

my $basename = basename($filename);
(my $utt_id = $basename) =~ s/\.[^.]+$//;

open IN, $filename or die "Could not open $filename";

my $max_time = 0;
while (<IN>) {
  chomp;
  my @A = split;
  if (int($A[1]/$frame_shift+0.5) > $max_time) { $max_time = int($A[1]/$frame_shift+0.5) }
}

my @vad = (1)x$max_time;

close IN;

open IN, $filename or die "Could not open $filename";

while (<IN>) {
  chomp;
  my @A = split;

  for (my $i = int($A[0]/$frame_shift); $i <= int($A[1]/$frame_shift+0.5); $i++) {
    $vad[$i] = 2;
  }
}

print STDOUT $utt_id;
foreach (@vad) {
  print STDOUT " $_";
}
print STDOUT "\n";
