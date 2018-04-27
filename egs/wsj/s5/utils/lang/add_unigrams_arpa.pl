#!/usr/bin/env perl

# Copyright 2018  Xiaohui Zhang
# Apache 2.0.
#
use strict;
use warnings;
use Getopt::Long;

my $Usage = <<EOU;
# This is a simple script to add unigrams to an ARPA lm file.
Usage: utils/lang/add_unigrams_arpa.pl [options] <oov-prob-file> <scale> <input-arpa >output-arpa
<oov-prob-file> contains a list of words and their probabilities, e.g. "jack 0.2". All probs will be
scaled by a positive scalar <scale> and then be used as the unigram prob. of the added word.
The scale should approximiately relect the OOV rate of the language in concern.
EOU

my @F;
my @OOVS;

if (@ARGV != 2) {
  die $Usage;
}

# Gets parameters.
my $oov_prob_file = shift @ARGV;
my $scale = shift @ARGV;
my $arpa_in = shift @ARGV;
my $arpa_out = shift @ARGV;

# Opens files.
open(F, "<$oov_prob_file") || die "$0: Fail to open $oov_prob_file\n";
while (<F>) { push @OOVS, $_; }
my $num_oovs = @OOVS;

$scale > 0.0 || die "Bad scale";
print STDERR "$0: Creating LM file with additional unigrams, using $oov_prob_file\n";

my %vocab;
my $unigram = 0;
my $num_unigrams = 0;
my @lines;

# Parse and record the head and unigrams in the ARPA LM.
while(<STDIN>) {
  if (m/^ngram 1=(\d+)/) { $num_unigrams = $1; }
  
  if (m/^\\2-grams:$/) { last; }
  if (m/^\\1-grams:$/) { $unigram = 1; push(@lines, $_); next; }
  if (m/^\\2-grams:$/) { $unigram = 0; }

  my @col = split(" ", $_);
  if ( $unigram == 1 ) {
    # Record in-vocab words into a map.
    if ( @col > 0 ) {
      my $word = $col[1];
      $vocab{$word} = 1;
      push(@lines, $_);
    } else {
      # Insert out-of-vocab words and their probs into the unigram list.
      foreach my $l (@OOVS) {
        my @A = split(" ", $l);
        @A == 2 || die "bad line in oov2prob: $_;";
        my $word = $A[0];
        my $prob = $A[1];
        if (exists($vocab{$word})) { next; }
        $num_unigrams ++;
        my $log10prob = (log($prob * $scale) / log(10.0));
        $vocab{$word} = 1;
        my $line = sprintf("%.6f\t$word\n", $log10prob);
        push(@lines, $line);
      }
    }
  } else { push(@lines, $_); }
}

# Print the head and unigrams, with the updated # unigrams in the head.
foreach my $l (@lines) {
  if ($l =~ m/ngram 1=/) {
    print "ngram 1=$num_unigrams\n";
  } else {
    print $l;
  }
}

# Print the left fields.
print "\n\\2-grams:\n";
while(<STDIN>) {
  print;
}

close(F);
exit 0
