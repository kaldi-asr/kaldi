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

# Converts the kws result with utterances into whole file offsets
use strict;
use warnings;
use utf8;
use Data::Dumper;
use Getopt::Long;

my $flen = 0.01;

GetOptions ("flen=f" => \$flen) or die "$0: Cannot parse command-line options\n";

my $segments=$ARGV[0];
my %SEGMENTS;

open(SEG, $segments) or die "Cannot open segment file $segments";
while(my $line = <SEG> ) {
  chomp $line;
  my @entries = split(" ", $line);
  die "The format of line \"$line\" does not conform the the segments file format" if @entries ne 4;

  $SEGMENTS{$entries[0]} = \@entries;
}


while (my $line = <STDIN> ) {
  chomp $line;
  my @entries = split(" ", $line);
  die "The format of line \"$line\" does not conform the result.* file format" if @entries ne 5;

  my $kw = $entries[0];
  my $utt = $entries[1];
  my $start = $entries[2];
  my $end = $entries[3];
  my $score = $entries[4];

  die "The utterance $utt is not in the segments file" unless exists $SEGMENTS{$utt};
  my $file = $SEGMENTS{$utt}->[1];
  my $utt_start = int( 0.5 + $SEGMENTS{$utt}->[2] / $flen);
  my $utt_end = int(0.5 + $SEGMENTS{$utt}->[3] / $flen);

  $start += $utt_start;
  $end += $utt_start;
  print "$kw $file $start $end $score\n";
}
