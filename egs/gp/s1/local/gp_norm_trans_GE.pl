#!/usr/bin/env perl
use warnings; #sed replacement for -w perl parameter

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


# This script normalizes the GlobalPhone German transcripts that have been 
# extracted in a format where each line contains an utterance ID followed by
# the transcript, e.g:
# GE008_10 man mag es drehen und wenden wie man will
# The normalization is similar to that in 'gp_format_dict_GE.pl' script.

my $usage = "Usage: gp_format_trans_GE.pl [-a] -i transcript > formatted\
Normalizes transcriptions for GlobalPhone German. The input format is assumed\
to be utterance ID followed by transcript on the same line.\
Options:\
  -a\tTreat acronyms differently (puts - between individual letters)\n";

use strict;
use Getopt::Long;
die "$usage" unless(@ARGV >= 1);
my ($acro, $in_trans);
GetOptions ("a" => \$acro,         # put - between letters of acronyms
            "i=s" => \$in_trans);  # Input transcription

open(T, "<$in_trans") or die "Cannot open transcription file '$in_trans': $!";
while (<T>) {
  s/\r//g;  # Since files could be in DOS format!
  s/\$//g;  # Some letters & acronyms written with $, e.g $A
  chomp;
  $_ =~ m:^(\S+)\s+(.+): || die "Bad line: $_";
  my $utt_id = $1;
  my $trans = $2;

  $trans =~ s/^\s*//; $trans =~ s/\s*$//;  # Normalize spaces
  $trans =~ s/ \,(.*?)\'/ $1/g;  # Remove quotation marks.
  $trans =~ s/ \-/ /g;  # conjoined noun markers, don't need them.

  print $utt_id;
  for my $word (split(/\s+/, $trans)) {
    # Distinguish acronyms before capitalizing everything, since they have 
    # different pronunciations. This may not be important.
    if (defined($acro)) {
      if ($word =~ /^[A-Z-]+(\-.*)*$/) {
	my @subwords = split('-', $word);
	$word = "";
	for my $i (0..$#subwords) {
	  if($subwords[$i] =~ /^[A-Z]{2,}$/) {
	    $subwords[$i] = join('-', split(//, $subwords[$i]));
	  }
	}
	$word = join('-', @subwords);
      }
    }
    $word =~ tr/a-z/A-Z/;  # Now, capitalize every word.
    print " $word"
  }
  print "\n";
}
