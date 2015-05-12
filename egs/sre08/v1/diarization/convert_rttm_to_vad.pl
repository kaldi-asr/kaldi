#!/usr/bin/perl -w
# Copyright 2015  Vimal Manohar (Johns Hopkins University)
# Apache 2.0.

use strict;
use POSIX;
use Getopt::Long;
use File::Basename;

my $frame_shift = 0.01;
my $ignore_boundaries = "false";
my $segments_file = "";

GetOptions('frame-shift:f' => \$frame_shift,
           'segments-out:s' => \$segments_file,
           'ignore-boundaries:s' => \$ignore_boundaries);

if (@ARGV > 1) {
  print STDERR "$0:\n" .
               "Usage: convert_rttm_to_vad.pl [options] [rttm] > <vad-out>\n";
  exit 1;
}

($frame_shift > 0.0001 && $frame_shift <= 1.0) ||
  die "Very strange frame-shift value '$frame_shift'";
($ignore_boundaries eq "false" || $ignore_boundaries eq "true") || 
  die "ignore-boundaries must be (true|false)";

if ($segments_file ne "") {
  open (SEGMENTS, ">", $segments_file) 
    or die "Cannot open $segments_file for writingn\n";
}

my %vad_for_file = ();
my %start_times = ();
my %end_times = ();

while (<>) {
  chomp;
  my @A = split;
  my $file = $A[1];

  if ($A[0] =~ m/SPKR-INFO/) {
    print STDERR "Reading RTTM for file $file\n";
    $vad_for_file{$file} = [];
    next;
  } elsif ($A[0] !~ m/SPEAKER/) {
    next;
  }
  my $start_time = floor($A[3] / $frame_shift);
  my $end_time = floor(($A[3] + $A[4]) / $frame_shift);

  if (! defined $start_times{$file}) {
    $start_times{$file} = $A[3];
    $end_times{$file} = $A[3] + $A[4];
  }

  $end_times{$file} = $A[3] + $A[4];

  exists $vad_for_file{$file} or die "SPKR-INFO not yet seen for file $file. RTTM is not sorted using rttmSort.pl?";

  for (my $i = scalar @{ $vad_for_file{$file} }; $i < $start_time; $i++) {
    $vad_for_file{$file}[$i] = 0;
  }
  scalar @{$vad_for_file{$file}} < $end_time or die "$end_time is < length of file";

  for (my $i = scalar @{$vad_for_file{$file}}; $i < $end_time; $i++) {
    $vad_for_file{$file}[$i] = 1;
  }
}

foreach (keys %vad_for_file) {
  my $file = $_;
  my $utt_id = $_;

  defined $start_times{$file} or die "Start time for $file not found\n";
  defined $end_times{$file} or die "End time for $file not found\n";

  my $start_time = floor($start_times{$file} / $frame_shift);
  my $end_time = floor($end_times{$file} / $frame_shift);
  if ($ignore_boundaries eq "true") {
    if ($segments_file ne "") {
      $utt_id = sprintf("$file-%06d-%06d", $start_time, $end_time);
      print SEGMENTS "$utt_id $file " . sprintf("%5.2f %5.2f", $start_time * $frame_shift, $end_time * $frame_shift) . "\n";
    }
    print STDOUT $utt_id . " " . join(" ", @{ $vad_for_file{$_} }[$start_time..($end_time-1)]) . "\n";
  } else {
    print STDOUT $utt_id . " " . join(" ", @{ $vad_for_file{$_} }) . "\n";
  }
}

