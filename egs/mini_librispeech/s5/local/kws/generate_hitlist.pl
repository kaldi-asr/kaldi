#!/usr/bin/env perl
#===============================================================================
# Copyright 2018  (Author: Yenda Trmal <jtrmal@gmail.com>)
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

# this will generate the hitlist (list of all hits) using the word-level
# alignments
# Format of the file
# utt-id word-1 duration-1 ; word-2 duration-2 ; ....
# it is exactly the same format that you can get from ali-to-phones with
# parameter --write-lengths (see the script create_hitlist.sh for complete
# example)

# The script is not very optimized -- the finding of the hits in the utterance
# is done by concatenating the word_ids sequence using '|' and then by searching
# for a substring processed the same way. After that, we workout the word-level
# indices of the individual hits (remember, there may be more hits per utterance)
# Probably still faster than rolling our own searching algorithm due to the fact
# that it goes directly to (optimized) perl's runtime function

use strict;
use warnings;
use utf8;

if ((scalar @ARGV > 2) || (scalar @ARGV < 1)) {
  print STDERR "Usage: $0 <keywords.int> [<alignment>]\n";
  print STDERR "E.g.\n";
  print STDERR "  $0 data/train_clean_5/kws/keywords.int < exp/tri3b_ali_train_clean_5/align.txt\n";
  die "Incorrect number of arguments."
}

my $keyword_file = shift @ARGV;
open(my $keywords, "<$keyword_file") or
  die "Cannot open $keyword_file for reading";

my @KW;
while (<$keywords>) {
  chomp;
  next unless $_;
  my @F = split;
  my $kwid = shift @F;
  push @KW, [$kwid, \@F];
}

while (<>) {
  chomp;
  next unless $_;

  my @F = split(" ", $_, 2);
  my $utt_id = shift @F;
  @F = split(/ ; /, $F[0]);

  my $frames_prev = 0;
  my @UTT;
  foreach my $entry (@F) {
    (my $word, my $frames) = split(" ", $entry, 2);
    if ($word ne 0) {
      my $frames_start = $frames_prev;
      my $frames_end = $frames_start + $frames;
      $frames_prev = $frames_end;
      push @UTT, [$word + 0, $frames_start, $frames_end];
    } else {
      $frames_prev += $frames;
    }
  }

  my $utt_string = '|' . join('|', map { $_->[0] } @UTT) . '|';
  my %utt_indices;
  my $counter = 0;
  my $idx = 0;
  #mapping between the position in the utt_string and the position of
  #the word in the original utterance
  while () {
    $idx = index($utt_string, '|', $idx);
    last if $idx == -1;
    $utt_indices{$idx} = $counter;
    $idx += 1;
    $counter +=1
  }


  foreach my $kw (@KW) {
    my $kw_string = "|" . join('|', @{$kw->[1]}) . '|';
    my $kwlen = scalar @{$kw->[1]};

    my $idx = 0;
    my @all_idx;
    while () {
      $idx = index($utt_string, $kw_string, $idx);
      last if $idx == -1;
      push @all_idx, $idx;
      $idx += 1;
    }

    foreach my $hit (@all_idx) {
      my $start_idx =  $utt_indices{$hit};
      my $end_idx = $start_idx + $kwlen - 1;
      my $start = $UTT[$start_idx]->[1];
      my $end = $UTT[$end_idx]->[2];

      print "$kw->[0] $utt_id $start $end 0\n";
    }
  }
}
