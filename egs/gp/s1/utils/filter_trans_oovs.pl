#!/usr/bin/perl -w

# Copyright 2012  Arnab Ghoshal

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


# This script reads a transcript from stanard input and a word list file (whose
# first field is considered only), and writes the transcript to the standard 
# output with all words not on the word list replaced by the argument to the 
# --map-oov option. 

my $usage = "Usage: filter_trans_oovs.pl -d word_list --map-oov symbol [--ignore-first-field] [transcript] > filtered\n
 Converts all words in the transcript that are not on the word-list to the OOV 
 symbol. --ignore-first-field will make it print the first field as is (e.g. if 
 it is the utterance ID).\n";

use strict;
use Getopt::Long;
die "$usage" unless(@ARGV >= 1);
my ($wlist_file, $oov_symbol, $ignore_first_field);
GetOptions ("d=s" => \$wlist_file,
            "--map-oov=s" => \$oov_symbol,
            "--ignore-first-field" => \$ignore_first_field);

die $usage unless(defined($wlist_file) && defined($oov_symbol));

open(W, "<$wlist_file") or die "Cannot open word list file '$wlist_file': $!";
my(%seen_words);
while (<W>) {
  m/^(\S+).*$/ or die "Bad line: '$_'";
  $seen_words{$1} = 1;
}
die "OOV symbol '$oov_symbol' must be present in word list." 
  unless (defined($seen_words{$oov_symbol}));

while (<>) {
  my @words = split;
  if (defined($ignore_first_field)) {
    print shift @words, " ";
  }
  warn "Found empty line." if (scalar(@words) == 0);
  my @buffer = ();
  for my $w (@words) {
    if (defined($seen_words{$w})) {
      push @buffer, $w;
    } else {
      push @buffer, $oov_symbol;
    }
  }
  print join(" ", @buffer), "\n";
}
