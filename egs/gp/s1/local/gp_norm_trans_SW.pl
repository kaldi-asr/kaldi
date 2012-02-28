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


# This script normalizes the GlobalPhone Swedish transcripts that have been 
# extracted in a format where each line contains an utterance ID followed by
# the transcript, e.g:
# SW002_6 sextio <#60> procent a^r enbilsaOkerier 
# The normalization is similar to that in 'gp_format_dict_SW.pl' script.

my $usage = "Usage: gp_format_trans_SW.pl -i transcript > formatted\
Normalizes transcriptions for GlobalPhone Swedish. The input format is \
assumed to be utterance ID followed by transcript on the same line.\n";

use strict;
use Getopt::Long;
die "$usage" unless(@ARGV >= 1);
my ($in_trans);
GetOptions ("i=s" => \$in_trans);  # Input transcription

open(T, "<$in_trans") or die "Cannot open transcription file '$in_trans': $!";
while (<T>) {
  s/\r//g;  # Since files could be in DOS format!
  chomp;
  $_ =~ m:^(\S+)\s+(.+): or die "Bad line: $_";
  my $utt_id = $1;
  my $trans = $2;

  $trans =~ s/^\s*//; $trans =~ s/\s*$//;  # Normalize spaces
  $trans =~ s/\`(.*?)\'/$1/g;  # Remove quotation marks.

  print $utt_id;
  for my $word (split(/\s+/, $trans)) {
    next if ($word =~ /\<\#.*\>/);  # Numbers are written as digits also.
    $word =~ tr/a-z/A-Z/;  # Now, capitalize every word.
    print " $word"
  }
  print "\n";
}
