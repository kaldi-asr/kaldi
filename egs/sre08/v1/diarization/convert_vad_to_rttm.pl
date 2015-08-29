#!/usr/bin/perl -w
# Copyright 2015  Vimal Manohar (Johns Hopkins University)
# Apache 2.0.

use strict;
use POSIX;
use Getopt::Long;
use File::Basename;

print STDERR join(@ARGV) . "\n";

my $frame_shift = 0.01;
my $segments = "";
my $speech_class = 2;
my $silence_class = 1;
GetOptions('frame-shift:f' => \$frame_shift,
           'segments:s' => \$segments,
           'speech-class:i' => \$speech_class,
           'silence-class:i' => \$silence_class) or die;

my $in;

if (@ARGV > 1) {
  print STDERR "$0:\n" .
               "Usage: convert_vad_to_rttm.pl [options] [<vad-file>] > <rttm-out>\n";
  exit 1;
}

if (@ARGV == 0) {
  $in = *STDIN;
} else {
  open $in, $ARGV[0] or die "Could not open $ARGV[0]";
}
($frame_shift > 0.0001 && $frame_shift <= 1.0) ||
  die "Very strange frame-shift value '$frame_shift'";

print STDERR "Extracting RTTM from VAD\n";

my %utt2file = ();
my %utt2start = ();
if ($segments ne "") {
  open SEGMENTS, $segments or die "Could not open segments file $segments\n";
  while (<SEGMENTS>) {
    chomp;
    my @F = split;
    (scalar @F == 4) or die "$0: Invalid line $_ in $segments\n";
    $utt2file{$F[0]} = $F[1];
    $utt2start{$F[0]} = $F[2];
  }
}

my %seen_files = ();
while (<$in>) {
  chomp;
  my @A = split;
  my $file_id = $A[0];
  if ($segments ne "") {
    (defined $utt2file{$A[0]}) or die "$0: Unknown utterance $A[0] in VAD\n";
    $file_id = $utt2file{$A[0]};
  } 
  if (! defined $seen_files{$file_id}) {
    print STDOUT "SPKR-INFO $file_id 1 <NA> <NA> <NA> unknown speech <NA>\n";
    $seen_files{$file_id} = 1;
  }

  my $state = 1;       # silence state
  my $begin_time = 0;
  my $end_time = 0;
  for (my $i = 1; $i < $#A; $i++) {
    if ($state == 1 && $A[$i] == $speech_class) { # speech start
      $begin_time = ($i-1) * $frame_shift;
      $state = 2;
    } elsif ($state == 2 && $A[$i] == $silence_class) { # silence start
      $end_time = ($i-1) * $frame_shift;
      $state = 1;
      my $dur = $end_time - $begin_time;
      if ($segments ne "") {
        $begin_time = $begin_time + $utt2start{$A[0]};
      }
      print STDOUT sprintf("SPEAKER $file_id 1 %5.2f %5.2f <NA> <NA> speech <NA>\n", $begin_time, $dur);
    } elsif ($A[$i] != $speech_class && $A[$i] != $silence_class) {
      die "Unknown class $A[$i]\n";
    }
  }
  if ($state == 2) {
    my $dur = ($#A-1)*$frame_shift - $begin_time;
    if ($segments ne "") {
      $begin_time = $begin_time + $utt2start{$A[0]};
    }
    print STDOUT sprintf("SPEAKER $file_id 1 %5.2f %5.2f <NA> <NA> speech <NA>\n", $begin_time, $dur);
  }
}
