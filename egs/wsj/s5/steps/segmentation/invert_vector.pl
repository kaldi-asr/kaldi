#! /usr/bin/perl

# Copyright 2016  Vimal Manohar
# Apache 2.0

use strict;
use warnings;

while (<STDIN>) {
  chomp;
  my @F = split;
  my $utt = shift @F;
  shift @F;

  print "$utt [ ";
  for (my $i = 0; $i < $#F; $i++) {
    if ($F[$i] == 0) {
      print "1 ";
    } else {
      print 1.0/$F[$i] . " ";
    }
  }
  print "]\n";
}
