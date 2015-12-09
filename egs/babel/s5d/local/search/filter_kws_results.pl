#!/usr/bin/env perl
#===============================================================================
# Copyright 2015  (Author: Yenda Trmal <jtrmal@gmail.com>)
# 
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
#===============================================================================

use strict;
use warnings;
use utf8;
use Data::Dumper;
use Getopt::Long;

# if parameter nbest > 0, then filters the result list so that there is no
# more than nbest hits in the output for each of the KWID
#

my $nbest = 250;
my $duptime = 50;

GetOptions ("nbest=i" => \$nbest);

# Function for sorting
sub KwslistOutputSort {
  if ($a->[0] ne $b->[0]) {
    if ($a->[0] =~ m/[0-9]+$/ && $b->[0] =~ m/[0-9]+$/) {
      ($a->[0] =~ /([0-9]*)$/)[0] <=> ($b->[0] =~ /([0-9]*)$/)[0]
    } else {
      $a->[0] cmp $b->[0];
    }
  } elsif ($a->[5] ne $b->[5]) {
    $b->[5] <=> $a->[5];
  } else {
    $a->[1] cmp $b->[1];
  }
}

sub KwslistDupSort {
  my ($a, $b, $duptime) = @_;
  if ($a->[1] ne $b->[1]) {
    $a->[1] cmp $b->[1];
  } elsif (abs($a->[2]-$b->[2]) >= $duptime){
    $a->[2] <=> $b->[2]; 
  } elsif ($a->[4] ne $b->[4]) {
    #score
    $a->[4] <=> $b->[4]; 
  } else {
    $b->[3] <=> $a->[3];
  }
}

my %RESULTS;
while ( my $line = <STDIN> ) {
  chomp $line;
  my @F = split " ", $line;
  @F == 5 || die "$0: Bad number of columns in raw results \"$line\"\n";
  push @{$RESULTS{$F[0]}}, \@F;
}

foreach my $kw (sort keys %RESULTS) {
  my @tmp = sort { KwslistDupSort($a, $b, $duptime) } @{$RESULTS{$kw}};
  my @results;
  
  @results = ();
  if (@tmp >= 1) {push(@results, $tmp[0])};
  for (my $i = 1; $i < scalar(@tmp); $i ++) {
    my $prev = $results[-1];
    my $curr = $tmp[$i];
    if ((abs($prev->[2]-$curr->[2]) < $duptime ) &&
        ($prev->[1] eq $curr->[1])) {
      next;
    } else {
      push(@results, $curr);
    }
  }

  @results = sort { ($b->[4] + 0.0) <=> ($a->[4] + 0.0) } @results;

  my $len = scalar @results < $nbest ? scalar @results : $nbest;
  for (my $i=0; $i < $len; $i++) {
    print join(" ", @{$results[$i]}) . "\n";
  }
}

