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

my $Usage=<<EOU;
The script normalizes the stream of categories information into one.

Usage: $0 <options> > categories
 e.g.: cat partial_categories.* | $0 > categories

Allowed options:
  --one-per-line    : by default, there will be only one line for each KWID
                      THis option changes the output format so that for
                      each pair "KWID CATEGORY" will be on a single line.

Note:
  Reads the stream of categories information in the format

  keyword-ID1 category category2
  keyword-ID2 category2
  keyword-ID1 category category2

  The duplicities are allowed (and will be removed).
  Multiple categories per line are allowed (and will be merged)

  The purpose of the script is to be able to merge the information from different
  scripts. Each script can generate it's own information about categories
  and this script can be then used to merge these partial tables into one global
EOU

use strict;
use warnings;
use utf8;
use Getopt::Long;
use Data::Dumper;
use POSIX;

my $one_per_line;

GetOptions("one-per-line", \$one_per_line) or
  do {
  print STDERR "Cannot parse the command-line parameters.\n";
  print STDERR "$Usage\n";
  die "Cannot continue\n"
};

if (@ARGV != 0) {
  print STDERR "Incorrect number of command-line parameters\n";
  print STDERR "$Usage\n";
  die "Cannot continue\n"
}

my %GROUPS;

while (my $line=<STDIN>) {
  chomp $line;
  my @entries = split " ", $line;

  die "The line \"$line\" does not have correct format" if @entries < 2;

  my $kwid=shift @entries;
  for my $category (@entries) {
    $GROUPS{$kwid}->{$category} = 1;
  }
}

for my $kwid (sort keys %GROUPS) {
  if ($one_per_line) {
    foreach my $category (sort keys %{$GROUPS{$kwid}} ) {
      print $kwid . " " . $category . "\n";
    }
  } else {
    print $kwid . " " . join(" ", sort keys %{$GROUPS{$kwid}}) . "\n";
  }
}
