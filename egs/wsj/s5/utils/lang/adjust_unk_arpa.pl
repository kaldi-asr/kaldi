#!/usr/bin/env perl

# Copyright 2018  Xiaohui Zhang
# Apache 2.0.
#
use strict;
use warnings;
use Getopt::Long;

my $Usage = <<EOU;
# This is a simple script to set/scale the unigram prob of the OOV dict entry in an ARPA lm file.
Usage: utils/lang/adjust_unk_arpa.pl [options] <oov-dict-entry> <unk-scale> <input-arpa >output-arpa

Allowed options:
  --fixed-value (true|false)   : If true, interpret the unk-scale as a fixed value we'll set to
                                 the unigram prob of the OOV dict entry, rather than using it to
                                 scale the unigram prob.
EOU

my $fixed_value = "false";
GetOptions('fixed-value=s' => \$fixed_value);

($fixed_value eq "true" || $fixed_value eq "false") ||
  die "$0: Bad value for option --fixed-value\n";

if (@ARGV != 2) {
  die $Usage;
}

# Gets parameters.
my $unk_word = shift @ARGV;
my $unk_scale = shift @ARGV;
my $arpa_in = shift @ARGV;
my $arpa_out = shift @ARGV;

$unk_scale > 0.0 || die "Bad unk_scale"; # this must be positive
if ( $fixed_value eq "true" ) {
  print STDERR "$0: Setting the unigram prob of $unk_word in LM file as $unk_scale.\n";
} else {
  print STDERR "$0: Scaling the unigram prob of $unk_word in LM file by $unk_scale.\n";
}

my $unigram = 0; # wether we are visiting the unigram field or not.

# Change the unigram prob of the unk-word in the ARPA LM.
while(<STDIN>) {
  if (m/^\\1-grams:$/) { $unigram = 1; }
  if (m/^\\2-grams:$/) { $unigram = 0; }
  my @col = split(" ", $_);
  if ( $unigram == 1 && @col > 1 && $col[1] eq $unk_word ) {
    if ( $fixed_value eq "true" ) {
      $col[0] = (log($unk_scale) / log(10.0));
    } else {
      $col[0] += (log($unk_scale) / log(10.0));
    }
    my $line = join("\t", @col);
    print "$line\n";
  } else {
    print;
  }
}

exit 0
