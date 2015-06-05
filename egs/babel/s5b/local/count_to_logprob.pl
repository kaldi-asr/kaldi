#!/usr/bin/env perl

# Copyright 2012  Johns Hopkins University (Author: Guoguo Chen)
# Apache 2.0.
#

use strict;
use warnings;
use Getopt::Long;

my $Usage = <<EOU;
Usage:    count_to_logprob.pl <confusing_in|-> <counfusing_out|->
          This script takes in the confusion phone pair counts and converts
          the counts into negated log probabilities. The counts should be in
          the following format:
          p1 p2 count1        // For substitution
          p3 <eps> count2     // For deletion
          <eps> p4 count3     // For insertion

Allowed options:
  --cutoff              : Minimal count to be considered                (int   , default=1)
EOU

my $cutoff = 1;
GetOptions('cutoff=i' => \$cutoff);

@ARGV == 2 || die $Usage;

# Workout the input and output parameters
my $cm_in = shift @ARGV;
my $cm_out = shift @ARGV;

open(I, "<$cm_in") || die "$0: Fail to open keywords file $cm_in\n";
open(O, ">$cm_out") || die "$0: Fail to write confusion matrix $cm_out\n";

# Collect counts
my %ins;
my %del;
my %subs;
my %phone_count;
my $ins_count = 0;
my $del_count = 0;
while (<I>) {
  chomp;
  my @col = split();
  @col == 3 || die "$0: Bad line in confusion matrix file: $_\n";
  my ($p1, $p2, $count) = ($col[0], $col[1], $col[2]);
  $count >= $cutoff || next;
  if ($p1 eq "<eps>" && $p2 ne "<eps>") {
    $ins{$p2} = $count;
    $ins_count += $count;
  } elsif ($p1 ne "<eps>" && $p2 eq "<eps>") {
    $del{$p1} = $count;
    $del_count += $count;
  } elsif ($p1 ne "<eps>" && $p2 ne "<eps>") {
    $p1 ne $p2 || next;           # Skip same phone convert
    $subs{"${p1}_$p2"} = $count;
    if (defined($phone_count{$p1})) {
      $phone_count{$p1} += $count;
    } else {
      $phone_count{$p1} = $count;
    }
  }
}

# Compute negated log probability
foreach my $key (keys %ins) {
  $ins{$key} = -log($ins{$key}/$ins_count);
}
foreach my $key (keys %del) {
  $del{$key} = -log($del{$key}/$del_count);
}
foreach my $key (keys %subs) {
  my @col = split(/_/, $key);
  $subs{$key} = -log($subs{$key}/$phone_count{$col[0]});
}

# print results
my $output = "";
foreach my $key (keys %ins) {
  $output .= "<eps> $key $ins{$key}\n";
}
foreach my $key (keys %del) {
  $output .= "$key <eps> $del{$key}\n";
}
foreach my $key (keys %subs) {
  my @col = split(/_/, $key);
  $output .= "$col[0] $col[1] $subs{$key}\n";
}

print O $output;

close(I);
close(O);
