#!/usr/bin/env perl

# Copyright 2016  Vimal Manohar
# Apache 2.0.

use warnings;

# This script reads from stdin a feats.scp file that contains frame ranges and
# ensures that they don't exceed the maximum number of frames supplied in the
# <utt2max-frames> file. 
# <utt2max-frames> is usually computed using get_utt2num_frames.sh on the 
# original directory which will be segmented using 
# utils/data/subsegment_data_dir.sh.
# 
# e.g. feats.scp
# utt_foo-1 foo-bar.ark:514231[721:892]
# 
# utt2max-frames
# utt_foo-1 891
# 
# fixed_feats.scp
# utt_foo-1 foo-bar.ark:514231[721:890]
# 
# Note: Here 891 is the number of frames in the archive foo-bar.ark
# The frame end for utt_foo-1, i.e. 892 (0-indexed) exceeds the archive size
# (891) by two frames. This script fixes that line by truncating the range 
# to 890.

if (scalar @ARGV != 1) {
  my $usage = <<END;
This script reads from stdin a feats.scp file that contains frame ranges and
ensures that they don't exceed the maximum number of frames supplied in the
<utt2max-frames> file. 

Usage: $0 <utt2max-frames> < feats.scp > fixed_feats.scp
END
  die "$usage";
}

my $utt2max_frames_file = $ARGV[0];

open MAX_FRAMES, $utt2max_frames_file or die "$0: Could not open file $utt2max_frames_file";

my %utt2max_frames;

while (<MAX_FRAMES>) {
  chomp;
  my @F = split;
  
  (scalar @F == 2) or die "$0: Invalid line $_ in $utt2max_frames_file";

  $utt2max_frames{$F[0]} = $F[1];
}

while (<STDIN>) {
  my $line = $_;
  
  #if (m/\[([^][]*)\]\[([^][]*)\]\s*$/) {
  #  print STDERR ("fix_subsegment_feats.pl: this script only supports single indices");
  #  exit(1);
  #}
  
  my $before_range = "";
  my $range = "";

  if (m/^(.*)\[([^][]*)\]\s*$/) {
    $before_range = $1;
    $range = $2;
  } else {
    print;
    next;
  }

  my @F = split(/ /, $before_range);
  my $utt = shift @F;
  defined $utt2max_frames{$utt} or die "fix_subsegment_feats.pl: Could not find key $utt in $utt2max_frames_file.\nError with line $line";

  if ($range !~ m/^(\d*):(\d*)([,]?.*)$/) {
    print STDERR "fix_subsegment_feats.pl: could not make sense of input line $_";
    exit(1);
  }
    
  my $row_start = $1;
  my $row_end = $2;
  my $col_range = $3;
  
  if ($row_start >= $utt2max_frames{$utt}) {
    print STDERR "Removing $utt because row_start $row_start >= file max length $utt2max_frames{$utt}\n";
    next;
  }  
  if ($row_end >= $utt2max_frames{$utt}) {
    print STDERR "Fixed row_end for $utt from $row_end to $utt2max_frames{$utt}-1\n";
    $row_end = $utt2max_frames{$utt} - 1;
  } 
   
  if ($row_start ne "") {
    $range = "$row_start:$row_end";
  } else {
    $range = "";
  }

  if ($col_range ne "") {
    $range .= ",$col_range";
  }
  print ("$utt " . join(" ", @F) . "[" . $range . "]\n");
}
