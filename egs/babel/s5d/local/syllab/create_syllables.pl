#!/usr/bin/env perl
#===============================================================================
# Copyright 2015 Johns Hopkins University (Author: Yenda Trmal<jtrmal@gmail.com>)
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
use Getopt::Long;
use Data::Dumper;

my $with_probs;
my $position_independent_phones;

GetOptions("with-probs" => \$with_probs,
  "position-independent-phones" => \$position_independent_phones
);

my %SYLLS;
my %LEXICON;

while (my $line = <STDIN>) {
  chomp $line;
  my $word; my $prob; my $pron;
  if ($with_probs) {
    ($word, $prob, $pron) = split(" ", $line, 3);
  } else {
    ($word, $pron) = split(" ", $line, 2);
  }
  my @syllabs = split(/\s*\t\s*/, $pron);

  my $pronlen= scalar @syllabs;
  my @extended_syllabs;
  if (( $syllabs[0] =~ /x\<.*\>/) || ($word eq "SIL")) {
    $SYLLS{$pron} +=1;
    push @extended_syllabs, $pron;
  } elsif ($pronlen == 1) {
    my $syl;
    my @phones=split " ", $syllabs[0];

    if ($position_independent_phones) {
      $syl = join(" ", @phones);
    } else {
      my @phones2 = map { $_ . "_I" } @phones;

      if (scalar(@phones)  == 1 ) {
        $syl = "$phones[0]_S";
      } else {
        $phones2[0] =  $phones[0] . "_B" unless $position_independent_phones;
        $phones2[-1] = $phones[-1] ."_E" unless $position_independent_phones;
        $syl = join(" ", @phones2);
      }
    }
    $SYLLS{$syl} += 1;
    push @extended_syllabs, $syl;
  } else {
    for (my $i = 0; $i lt $pronlen; $i+=1) {
      my $syl;
      my @phones=split " ", $syllabs[$i];
      my $first_index = 0;
      my $last_index = scalar(@phones)-1;

      if ($position_independent_phones) {
        $syl = join(" ", @phones);
      } else {
        my @phones2 = map { $_ . "_I" } @phones;

        if ($i == 0) {
          $phones2[$first_index] = $phones[$first_index] . "_B";
        } elsif ( $i == ($pronlen - 1)) {
          $phones2[$last_index] = $phones[$last_index] . "_E";
        }
        $syl = join(" ", @phones2);
      }

      push @extended_syllabs, $syl;
      $SYLLS{$syl} += 1;
    }
  }
  push @{$LEXICON{$word}}, \@extended_syllabs;
}


my %VOCAB;
my %COUNTS;
my %REV_VOCAB;
foreach my $syl (keys %SYLLS) {
  my $seq=1;
  my $word=$syl;
  $word =~ s/_[^\s]*//g;
  $word =~ s/ //g;
  $word =~ s/[^a-zA-Z0-9<>-|\/]//g;

  my $wordx=$word;
  $wordx .= "#$seq";
  while (exists $COUNTS{$wordx}) {
    $seq += 1;
    $wordx = "$word#$seq";
  }

  $COUNTS{$wordx} += $SYLLS{$syl};
  push @{$VOCAB{$wordx}}, $syl;
  $REV_VOCAB{$syl} = $wordx;
}

open(my $lex_f, "|sort -u >  $ARGV[0]") or
die "Cannot open the file\"$ARGV[0]\" for writing";

foreach my $word (keys %VOCAB) {
  print $lex_f "$word\t" . join("\t", @{$VOCAB{$word}}) . "\n";
}

close($lex_f);

open(my $word2syll_f, "|sort -u > $ARGV[1]") or
die "Cannot open the file\"$ARGV[1]\" for writing";

foreach my $word (keys %LEXICON) {
  foreach my $pron (@{$LEXICON{$word}}) {
    my @pron_in_syllabs;
    foreach my $syl (@{$pron}) {
      die "In word $word, pronunciation $pron: syllable $syl not in the lexicon!" unless exists $REV_VOCAB{$syl};
      push @pron_in_syllabs, $REV_VOCAB{$syl};
    }
    print $word2syll_f "$word\t" . join(" ", @pron_in_syllabs) . "\n";
  }
}

close($word2syll_f);

open(my $word2ali_f, "|sort -u > $ARGV[2]") or
die "Cannot open the file\"$ARGV[2]\" for writing";

foreach my $word (keys %LEXICON) {
  foreach my $pron (@{$LEXICON{$word}}) {
    print $word2ali_f "$word\t$word\t" . join(" ", @{$pron}) . "\n";
  }
}

close($word2ali_f);

