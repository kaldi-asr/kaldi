#! /usr/bin/perl
use strict;
use warnings;

# This script parses a features config file such as conf/mfcc.conf
# and returns the pair of values frame_shift and frame_overlap in seconds.

my $frame_shift = 0.01;
my $frame_overlap = 0.015;

while (<>) {
  if (m/--frame-length=(\d+)/) {
    $frame_shift = $1 / 1000;
  } 

  if (m/--window-length=(\d+)/) {
    $frame_overlap = $1 / 1000 - $frame_shift;
  }
}

print "$frame_shift $frame_overlap\n";
