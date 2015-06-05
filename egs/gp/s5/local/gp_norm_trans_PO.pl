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


# This script normalizes the GlobalPhone Portuguese transcripts that have been 
# extracted in a format where each line contains an utterance ID followed by
# the transcript, e.g:
# PO001_64 o grupo na~o mencionou ataques a bomba
# The normalization is similar to that in 'gp_norm_dict_PO.pl' script.

my $usage = "Usage: gp_norm_trans_PO.pl -i transcript > formatted\
Normalizes transcriptions for GlobalPhone Portuguese. The input format is \
assumed to be utterance ID followed by transcript on the same line.\n";

use strict;
use Getopt::Long;
use Unicode::Normalize;

die "$usage" unless(@ARGV >= 1);
my ($in_trans, $keep_rmn, $uppercase);
GetOptions ("r" => \$keep_rmn,     # keep words in GlobalPhone-style ASCII (rmn)
	    "u" => \$uppercase,    # convert words to uppercase
	    "i=s" => \$in_trans);  # Input transcription

binmode(STDOUT, ":encoding(utf8)") unless (defined($keep_rmn));

open(T, "<$in_trans") or die "Cannot open transcription file '$in_trans': $!";
while (<T>) {
  s/\r//g;  # Since files may have CRLF line-breaks!
  chomp;
  $_ =~ m:^(\S+)\s+(.+): or die "Bad line: $_";
  my $utt_id = $1;
  my $trans = $2;

  $trans =~ s/\`(.*?)\'/$1/g;  # Remove quotation marks.
  $trans =~ s/</ /g;  # Fragments are enclosed in < & > : space them properly
  $trans =~ s/\-?>/ /g;

  $trans = &rmn2utf8_PO($trans) unless (defined($keep_rmn));
  if (defined($uppercase)) {
    $trans = uc($trans);
  } else {
    $trans = lc($trans);
  }

  # Normalize spaces
  $trans =~ s/^\s*//; $trans =~ s/\s*$//; $trans =~ s/\s+/ /g;
  print "$utt_id $trans\n";
}


sub rmn2utf8_PO {
  my ($in_str) = "@_";
  
  $in_str =~ s/A\:/\x{00C0}/g;
  $in_str =~ s/A\+/\x{00C1}/g;
  $in_str =~ s/A\^/\x{00C2}/g;
  $in_str =~ s/A\~/\x{00C3}/g;
  $in_str =~ s/C\:/\x{00C7}/g;
  $in_str =~ s/E\+/\x{00C9}/g;
  $in_str =~ s/E\^/\x{00CA}/g;
  $in_str =~ s/I\+/\x{00CD}/g;
  $in_str =~ s/N\~/\x{00D1}/g;
  $in_str =~ s/O\+/\x{00D3}/g;
  $in_str =~ s/O\^/\x{00D4}/g;
  $in_str =~ s/O\~/\x{00D5}/g;
  $in_str =~ s/U\+/\x{00DA}/g;
  $in_str =~ s/U\^/\x{00DC}/g;

  $in_str =~ s/a\:/\x{00E0}/g;
  $in_str =~ s/a\+/\x{00E1}/g;
  $in_str =~ s/a\^/\x{00E2}/g;
  $in_str =~ s/a\~/\x{00E3}/g;
  $in_str =~ s/c\:/\x{00E7}/g;
  $in_str =~ s/e\+/\x{00E9}/g;
  $in_str =~ s/e\^/\x{00EA}/g;
  $in_str =~ s/i\+/\x{00ED}/g;
  $in_str =~ s/n\~/\x{00F1}/g;
  $in_str =~ s/o\+/\x{00F3}/g;
  $in_str =~ s/o\^/\x{00F4}/g;
  $in_str =~ s/o\~/\x{00F5}/g;
  $in_str =~ s/u\+/\x{00FA}/g;
  $in_str =~ s/u\^/\x{00FC}/g;

  return NFC($in_str);  # recompose & reorder canonically
}
