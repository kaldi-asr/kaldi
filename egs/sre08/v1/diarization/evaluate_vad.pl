#!/usr/bin/perl -w
# Copyright 2015  Vimal Manohar (Johns Hopkins University)
# Apache 2.0.

use strict;
use Getopt::Long;

if (@ARGV != 2) {
  print STDERR "$0:\n" .
               "Usage: evaluate_vad.pl [options] <ref-file> <hyp-file>\n";
  exit 1;
}

my $ref_file = $ARGV[0];
my $hyp_file = $ARGV[1];

open REF, $ref_file or die "$0: Unable to open reference vad $ref_file\n";

open HYP, $hyp_file or die "$0: Unable to open hypothesis vad $hyp_file\n";

my %hyps = ();

while (<HYP>) {
  chomp;
  my @A = split;
  $hyps{$A[0]} = [ @A[1..$#A] ];
#%  print STDERR join(' ', @{$hyps{$A[0]}}) . "\n";
}

#foreach (keys %hyps) {
#  print STDERR $_ . join(' ', @{$hyps{$_}}) . "\n";
#}

while (<REF>) {
  chomp;
  my @A = split;
  my $fp = 0;
  my $fn = 0;
  my $cor = 0;

  my @B = @A[1..$#A];

  my $i = 1;
  my @H = @{$hyps{$A[0]}};

  for ($i=0; $i <= $#B; $i++) {
    if ( ($B[$i] == 1) && ($i <= $#H) && ($H[$i] == 2) ) {
      $fp++;
    } elsif ( ($B[$i] == 2) && ( ($i > $#H) || ($H[$i] == 1) ) ) {
      $fn++;
    } 
    else {
      $cor++;
    }
  }
  while ($i <= $#H) {
    if ( $H[$i] == 2 ) {
      $fp++;
    } else {
      $cor++;
    }
    $i++;
  }

  my $n = scalar @B;
  print STDOUT $A[0] . " " . scalar @H . " " . scalar @B . " $cor $fp $fn " . sprintf("%6.4f %6.4f %6.4f\n", $cor/$n, $fp/$n, $fn/$n);
}
