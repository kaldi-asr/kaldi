#!/usr/bin/env perl
#
# Copyright 2015  University of Sheffield (Author: Ning Ma)
# Apache 2.0.
#
# Create transcriptions for the CHIME/GRID corpus from a list of
# file names (used as UTTERANCE-ID, e.g. s1_bgab3n)
# It outputs lines containing UTTERANCE-ID TRANSCRIPTIONS, e.g. 
#   s1_bgab3n BIN GREEN AT B THREE NOW
#
# Usage: create_chime1_trans.pl train.flist

use strict;
use warnings;

# Define silence label at begin/end of an utternace
my $sil = "<SIL>";

my $in_list = $ARGV[0];

open my $info, $in_list or die "could not open $in_list: $!";

while (my $line = <$info>) {
  chomp($line);
  $line =~ s/\.[^.]+$//; # Remove extension just in case
  my @tokens = split("_", $line); 
  my @chars = split("", $tokens[1]);
  my $trans;

  if ($chars[0] eq "b") { $trans = "BIN"}
  elsif ($chars[0] eq "l") { $trans = "LAY" }
  elsif ($chars[0] eq "p") { $trans = "PLACE" }
  elsif ($chars[0] eq "s") { $trans = "SET" }
  else { $trans = "!UNKNOWN"}

  if ($chars[1] eq "b") { $trans = $trans . " BLUE" }
  elsif ($chars[1] eq "g") { $trans = $trans . " GREEN" }
  elsif ($chars[1] eq "r") { $trans = $trans . " RED" }
  elsif ($chars[1] eq "w") { $trans = $trans . " WHITE" }
  else { $trans = $trans . "!UNKNOWN"}

  if ($chars[2] eq "a") { $trans = $trans . " AT" }
  elsif ($chars[2] eq "b") { $trans = $trans . " BY" }
  elsif ($chars[2] eq "i") { $trans = $trans . " IN" }
  elsif ($chars[2] eq "w") { $trans = $trans . " WITH" }
  else { $trans = $trans . "!UNKNOWN"}
  
  $trans = $trans . " " . uc($chars[3]);

  if ($chars[4] eq "z") { $trans = $trans . " ZERO" }
  elsif ($chars[4] eq "1") { $trans = $trans . " ONE" }
  elsif ($chars[4] eq "2") { $trans = $trans . " TWO" }
  elsif ($chars[4] eq "3") { $trans = $trans . " THREE" }
  elsif ($chars[4] eq "4") { $trans = $trans . " FOUR" }
  elsif ($chars[4] eq "5") { $trans = $trans . " FIVE" }
  elsif ($chars[4] eq "6") { $trans = $trans . " SIX" }
  elsif ($chars[4] eq "7") { $trans = $trans . " SEVEN" }
  elsif ($chars[4] eq "8") { $trans = $trans . " EIGHT" }
  elsif ($chars[4] eq "9") { $trans = $trans . " NINE" }
  else { $trans = $trans . "!UNKNOWN"}
  
  if ($chars[5] eq "a") { $trans = $trans . " AGAIN" }
  elsif ($chars[5] eq "n") { $trans = $trans . " NOW" }
  elsif ($chars[5] eq "p") { $trans = $trans . " PLEASE" }
  elsif ($chars[5] eq "s") { $trans = $trans . " SOON" }
  else { $trans = $trans . "!UNKNOWN"}
  
  #print "$line    $sil $trans $sil\n";
  print "$line\t$trans\n";
}

