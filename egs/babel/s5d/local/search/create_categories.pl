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
Parses the data file and outputs the basic KW categories

Usage:  $0 <source-file>
 e.g.:  $0 keywords.txt
    or  $0 --results results

Allowed options:
  --results          : instead of keyword specification format, keyword search
                       results format is assumed.

NOTE:
  If you need both information, you can call the script twice (with different
  parameters) and call local/search/normalize_categories.pl to merge (and normalize)
  these two tables together.
EOU

use strict;
use warnings;
use utf8;
use POSIX;
use Data::Dumper;
use Getopt::Long;
use open qw(:std :utf8);

binmode STDIN, ":utf8";
binmode STDOUT, ":utf8";
binmode STDERR, ":utf8";

my $result_format;
GetOptions("results", \$result_format) or do {
  print STDERR "Cannot parse the command-line parameters.\n";
  print STDERR "$Usage\n";
  die "Cannot continue\n"
};

if ( @ARGV > 1 ) {
  print STDERR "Incorrect number of command-line parameters\n";
  print STDERR "$Usage\n";
  die "Cannot continue\n"
}

sub QuantizeCount {
  my $count = shift @_;

  if ($count <= 0) {
    return "0";
  } elsif ($count == 1) {
    return "000-001";
  } elsif ($count <= 5) {
    return "002-005";
  } elsif ($count <=10) {
    return "006-010";
  } elsif ($count <=20) {
    return "011-020";
  } elsif ($count <=100) {
    return "021-100";
  } else {
    return "101-inf";
  }
}

if (not $result_format ) {
  my $kwlist_name=$ARGV[0];
  while (my $line = <>) {
    chomp $line;
    my ($kwid, $text) = split " ", $line, 2;

    my @words = split " ", $text;
    printf "$kwid NGramOrder=%03d\n", scalar @words;
    printf "$kwid Characters=%03d\n", length(join("", @words));
    print "$kwid $kwid\n";
  }
} else {
  my $prev_kwid = "";
  my $count = 0;

  while (my $line = <>) {
    chomp $line;
    my @entries = split " ", $line;
    next unless @entries;

    if ($prev_kwid ne $entries[0]) {
      if ($prev_kwid) {
        print "$prev_kwid ResCount=$count\n";
        print "$prev_kwid ResCountQuant=" . QuantizeCount($count) . "\n";
      }
      $count = 0;
      $prev_kwid = $entries[0];
    }
    $count += 1;
  }
}


