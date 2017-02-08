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

my $Usage = <<EOU;
Filters the kws results file and remove duplicates and/or prints out only given
number of results with the best score (for each KWID individually).

Usage: cat results | $0 <options>  > output
 e.g.:  gunzip -c exp/tri5/kws/result.*.gz | $0 > exp/tri5/kws/results

Allowed options:
  --nbest           :  how many best results (for each KWID) should be printed
                       (int, default -1, i.e. no limit)
  --duptime         :  duplicates detection, tolerance (in frames) for being
                       the same hits (int,  default = 50)
  --likes
  --probs

CAVEATS:
  The script tries to be  memory-effective. The impact of this is that we
  assume the results are sorted by KWID (i.e. all entries with the same KWID
  are in a continuous block). The user is responsible for sorting it.
EOU

use strict;
use warnings;
use utf8;
use POSIX;
use Data::Dumper;
use Getopt::Long;

# if parameter nbest > 0, then filters the result list so that there is no
# more than nbest hits in the output for each of the KWID
#

my $nbest = -1;
my $duptime = 50;
my $likes = 0;

#print STDERR join(" ", $0, @ARGV) . "\n";
GetOptions ("nbest=f" => \$nbest,
            "likes" => \$likes,
            "probs" => sub{ $likes = 0},
            "duptime=i" => \$duptime) ||  do {
  print STDERR "Cannot parse the command-line parameters.\n";
  print STDERR "$Usage\n";
  die "Cannot continue\n"
};

if (@ARGV != 0) {
  print STDERR "Incorrect number of command-line parameters\n";
  print STDERR "$Usage\n";
  die "Cannot continue\n"
}

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
    #file
    $a->[1] cmp $b->[1];
  } elsif (abs($a->[2]-$b->[2]) >= $duptime){
    #start
    $a->[2] <=> $b->[2];
  } elsif ($a->[4] ne $b->[4]) {
    #score
    $b->[4] <=> $a->[4];
  } else {
    #end time
    $b->[3] <=> $a->[3];
  }
}

my @RESULTS;
my %SEEN_KWS;
my $kw = "";

while ( my $line = <STDIN> ) {
  chomp $line;
  my @F = split " ", $line;
  @F == 5 || die "$0: Bad number of columns in raw results \"$line\"\n";

  $F[4] = -$F[4] if $likes;

  if ($F[0] eq $kw) {
    push @RESULTS, \@F;
  } elsif ($kw eq "" ) {
    @RESULTS = ();
    push @RESULTS, \@F;
    $kw = $F[0];
  } else {

    my @results;
    my @tmp = sort { KwslistDupSort($a, $b, $duptime) } @RESULTS;

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

    # this is probably needed only when nbest > 0
    @results = sort { ($b->[4] + 0.0) <=> ($a->[4] + 0.0) } @results;

    my $len;
    if( $nbest > 0)  {
      $len = scalar @results < $nbest ? scalar @results : $nbest;
    } else {
      $len = scalar @results;
    }
    for (my $i=0; $i < $len; $i++) {
      $results[$i]->[4] = -$results[$i]->[4] if $likes;
      print join(" ", @{$results[$i]}) . "\n";
    }

    @RESULTS = ();
    push @RESULTS, \@F;
    $kw = $F[0];
  }
}
do {
  my @results;
  my @tmp = sort { KwslistDupSort($a, $b, $duptime) } @RESULTS;

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

  # this is probably needed only when nbest > 0
  @results = sort { ($b->[4] + 0.0) <=> ($a->[4] + 0.0) } @results;

  my $len;
  if( $nbest > 0)  {
    $len = scalar @results < $nbest ? scalar @results : $nbest;
  } else {
    $len = scalar @results;
  }
  for (my $i=0; $i < $len; $i++) {
    $results[$i]->[4] = -$results[$i]->[4] if $likes;
    print join(" ", @{$results[$i]}) . "\n";
  }
}


