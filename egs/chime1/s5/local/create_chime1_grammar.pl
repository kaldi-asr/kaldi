#!/usr/bin/env perl
#
# Copyright 2015  University of Sheffield (Author: Ning Ma)
# Apache 2.0.
#
# Prepare a simple grammar G.fst for the GRID corpus (CHiME 1/2)
# with silence at the beginning and the end of each utterance.
#

use strict;
use warnings;

# GRID has the following grammar:
# verb=bin|lay|place|set
# colour=blue|green|red|white
# prep=at|by|in|with
# letter=a|b|c|d|e|f|g|h|i|j|k|l|m|n|o|p|q|r|s|t|u|v|x|y|z
# digit=zero|one|two|three|four|five|six|seven|eight|nine
# coda=again|now|please|soon
# sil $verb $colour $prep $letter $digit $coda sil

my $state = 0;
my $state2 = $state + 1;
#my $sil = "<SIL>";
#print "$state $state2 $sil $sil 0.0\n";

#$state++;
#$state2 = $state + 1;
my @words = ("BIN", "LAY", "PLACE", "SET");
my $nWords = @words;
my $penalty = -log(1.0/$nWords);
foreach (@words) { print "$state $state2 $_ $_ $penalty\n"; }

$state++;
$state2 = $state + 1;
@words = ("BLUE", "GREEN", "RED", "WHITE");
$nWords = @words;
$penalty = -log(1.0/$nWords);
foreach (@words) { print "$state $state2 $_ $_ $penalty\n"; }

$state++;
$state2 = $state + 1;
@words = ("AT", "BY", "IN", "WITH");
$nWords = @words;
$penalty = -log(1.0/$nWords);
foreach (@words) { print "$state $state2 $_ $_ $penalty\n"; }

$state++;
$state2 = $state + 1;
@words = ("A".."V", "X", "Y", "Z");
$nWords = @words;
$penalty = -log(1.0/$nWords);
foreach (@words) { print "$state $state2 $_ $_ $penalty\n"; }

$state++;
$state2 = $state + 1;
@words = ("ZERO", "ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN", "EIGHT", "NINE");
$nWords = @words;
$penalty = -log(1.0/$nWords);
foreach (@words) { print "$state $state2 $_ $_ $penalty\n"; }

$state++;
$state2 = $state + 1;
@words = ("AGAIN", "NOW", "PLEASE", "SOON");
$nWords = @words;
$penalty = -log(1.0/$nWords);
foreach (@words) { print "$state $state2 $_ $_ $penalty\n"; }

#$state++;
#$state2 = $state + 1;
#print "$state $state2 $sil $sil 0.0\n";

print "$state2 0.0\n";

