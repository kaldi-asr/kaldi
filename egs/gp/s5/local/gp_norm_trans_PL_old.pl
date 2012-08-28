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


# This script normalizes the GlobalPhone Polish transcripts that have been 
# extracted in a format where each line contains an utterance ID followed by
# the transcript, e.g:
# PO007_8 Strach pierwszy - Unia nas wynarodowi
# The normalization is similar to that in 'gp_format_dict_PL.pl' script.

my $usage = "Usage: gp_norm_trans_PL.pl -i transcript > formatted\
Normalizes transcriptions for GlobalPhone Polish. The input format is \
assumed to be utterance ID followed by transcript on the same line.\n";

use strict;
use Getopt::Long;
use Unicode::Normalize;
use open ':encoding(utf8)';

binmode(STDOUT, ":encoding(utf8)");

die "$usage" unless(@ARGV >= 1);
my ($in_trans, $uppercase);
GetOptions ("u" => \$uppercase,    # convert words to uppercase
	    "i=s" => \$in_trans);  # Input transcription

open(T, "<$in_trans") or die "Cannot open transcription file '$in_trans': $!";
while (<T>) {
  $_ = NFD($_);  # UTF8 decompose
  s/\r//g;  # Since files could have CRLF line-breaks!
  chomp;
  $_ =~ m:^(\S+)\s+(.+): or die "Bad line: $_";
  my $utt_id = $1;
  my $trans = $2;

  $trans =~ s/\"/ /g;  # Remove quotation marks.
  $trans =~ s/[\,\.\?\!\:]/ /g;
  $trans =~ s/(\- | \-)/ /g;
  $trans =~ s/\x{FEFF}/ /g;  # zero-width space character!
  # Normalize spaces
  $trans =~ s/^\s*//; $trans =~ s/\s*$//; $trans =~ s/\s+/ /g;

  if (defined($uppercase)) {
    $trans = uc($trans);
  } else {
    $trans = lc($trans);
  }

  print "$utt_id $trans\n";
}
