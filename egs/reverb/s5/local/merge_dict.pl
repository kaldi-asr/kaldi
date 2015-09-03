#!/usr/bin/env perl
# Copyright 2013 MERL (author: Felix Weninger)

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.

use strict;

if ($#ARGV < 2) {
  print "Usage: $0 <d1> <d2> <target>\n";
  exit 1;
}

my ($dict1, $dict2, $target) = @ARGV;

open(D1, $dict1) or die "$dict1: $!";
open(D2, $dict2) or die "$dict2: $!";
open(T, '>', $target) or die "$target: $!";

# read all pronunciations from d1
my %d;

while (<D1>) {
  next if (/^#/);
  chomp;
  my @els = split(/\s+/, $_, 2);
  push(@{$d{$els[0]}}, uc $els[1]);
}

while (<D2>) {
  next if (/^;;/);
  chomp;
  my @els = split(/\s+/, $_, 2);
  $els[0] =~ s/\(\d+\)//;
  if (!defined $d{$els[0]}) {
    #print "Adding pronunciation from d2 for $els[0]\n";
    my $ptmp = $els[1];
    $ptmp =~ s/[0-2]//g;
    push(@{$d{$els[0]}}, $ptmp);
  }
}

for my $w (sort keys %d) {
  for my $p (0..$#{$d{$w}}) {
    print T "$w  $d{$w}[$p]\n"; #, join(" ", @{$d{$w}}), "\n";
  }
}
