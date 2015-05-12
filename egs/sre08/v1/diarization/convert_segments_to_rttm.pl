#!/usr/bin/perl -w
# Copyright 2015  Vimal Manohar (Johns Hopkins University)
# Apache 2.0.

use strict;
use POSIX;
use Getopt::Long;
use File::Basename;

print STDERR join(@ARGV) . "\n";

my $frame_shift = 0.01;
GetOptions('frame-shift:f' => \$frame_shift);

if (@ARGV != 1) {
  print STDERR "$0:\n" .
               "Usage: cat <segments-file> | convert_vad_to_rttm.pl [options] <uem-file> > <rttm-out>\n";
  exit 1;
}
open (UEM, ">", $ARGV[0]) or die "Could not open $ARGV[0]";

($frame_shift > 0.0001 && $frame_shift <= 1.0) ||
  die "Very strange frame-shift value '$frame_shift'";

print STDERR "Extracting RTTM from segments\n";

my %seen_files = ();
my %min_time = ();
my %max_time = ();

while (<STDIN>) {
  chomp;
  my @A = split;
  my $utt_id = $A[0];
  my $file_id = $A[1];
  
  if (! defined $seen_files{$file_id}) {
    print STDOUT "SPKR-INFO $file_id 1 <NA> <NA> <NA> unknown speech <NA>\n";
    $seen_files{$file_id} = 1;
    $min_time{$file_id} = $A[2];
  }

  print STDOUT sprintf("SPEAKER $file_id 1 %5.2f %5.2f <NA> <NA> speech <NA>\n", $A[2], $A[3] - $A[2]);
  $max_time{$file_id} = $A[3];
}

foreach (keys %min_time) {
  print UEM sprintf("$_ 1 %5.2f %5.2f\n", $min_time{$_}, $max_time{$_});
}

