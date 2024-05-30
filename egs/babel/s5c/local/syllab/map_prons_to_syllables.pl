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
use GetOpt::Long;

my $probs;

GetOptions ("--with-probs" => \$probs)

my $syllab_lexicon=$ARGV[0];

my %PRON2SYL;


open(my $f, $syllab_lexicon) or die "Cannot open file $syllab_lexicon\n";
while (my $line = <$f>) {
  chomp $line;

  my $syll;
  my $pron;
  my $prob;

  if ($probs) {
    $syll, $prob, $pron = split " ", $line, 3;
  } else {
    $syll, $pron = split " ", $line, 2;
  }
  $PRON2SYL{$pron} = $syll;
}

while (my $line = <STDIN>) {
  chomp $line;
  my ($word, $pron) = split(/\s*\t\s*/, $line, 2);
  my @syllabs = split(/\s*\t\s*/, $pron);

  my @syl_pron;
  foreach my $syl (@syllabs) {
    die "in $line unknown syllable $syl" unless exists $PRON2SYL{$syl};
    push @syl_pron, $PRON2SYL{$syl};
  }
  print "$word\t" . join(" ", @syl_pron) . "\n";

}
