#!/usr/bin/env perl
use warnings; #sed replacement for -w perl parameter

# Copyright 2012  Milos Janda;  Arnab Ghoshal

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
# PO007_8 Aktualny sekretarz generalny PS Franc2ois Hollande pozyskal1 jednak wie~kszos1c1
# The normalization is similar to that in 'gp_norm_dict_PL.pl' script.

my $usage = "Usage: gp_norm_trans_PL.pl -i transcript > formatted\
Normalizes transcriptions for GlobalPhone Polish. The input format is \
assumed to be utterance ID followed by transcript on the same line.\n";

use strict;
use Getopt::Long;
use Unicode::Normalize;
use open ':encoding(utf8)';

binmode(STDOUT, ":encoding(utf8)");

die "$usage" unless(@ARGV >= 1);
my ($in_trans, $keep_rmn, $uppercase);
GetOptions ("r" => \$keep_rmn,     # keep words in GlobalPhone-style ASCII (rmn)
	    "u" => \$uppercase,    # convert words to uppercase
	    "i=s" => \$in_trans);  # Input transcription

open(T, "<$in_trans") or die "Cannot open transcription file '$in_trans': $!";
while (<T>) {
  s/\r//g;  # Since files could have CRLF line-breaks!
  chomp;
  $_ =~ m:^(\S+)\s+(.+): or die "Bad line: $_";
  my $utt_id = $1;
  my $trans = $2;

  $trans =~ s/\"/ /g;  # Remove quotation marks.
  $trans =~ s/[\,\.\?\!\:\;\)\(\`]/ /g;
  $trans =~ s/(\- | \-)/ /g;
  $trans =~ s/\x{FEFF}/ /g;         # zero-width space character!
  $trans =~ s/\x{00AB}\x{00BB}//g;  # DOUBLE ANGLE QUOTATION MARKS
  # Normalize spaces
  $trans =~ s/^\s*//; $trans =~ s/\s*$//; $trans =~ s/\s+/ /g;

  $trans = &rmn2utf8_PL($trans) unless (defined($keep_rmn));  
  if (defined($uppercase)) {
    $trans = uc($trans);
  } else {
    $trans = lc($trans);
  }

  print "$utt_id $trans\n";
}


sub rmn2utf8_PL {
  my ($in_str) = "@_";

  # CAPITAL LETTERS
  $in_str =~ s/A\~/\x{0104}/g; # A WITH OGONEK
  $in_str =~ s/E\~/\x{0118}/g; # E WITH OGONEK

  $in_str =~ s/Z0/\x{017B}/g; # Z WITH DOT ABOVE

  $in_str =~ s/C1/\x{0106}/g; # LETTERS WITH ACUTE
  $in_str =~ s/L1/\x{0141}/g;
  $in_str =~ s/N1/\x{0143}/g;
  $in_str =~ s/O1/\x{00D3}/g;
  $in_str =~ s/S1/\x{015A}/g;
  $in_str =~ s/Z1/\x{0179}/g;

  $in_str =~ s/O2/\x{00D6}/g; # O WITH DIAERESIS (German umlaut)
  $in_str =~ s/U2/\x{00DC}/g; # U WITH DIAERESIS (German umlaut)
  $in_str =~ s/C2/\x{00C7}/g; # C WITH CEDILLA (from French)


  # SMALL LETTERS
  $in_str =~ s/a\~/\x{0105}/g;
  $in_str =~ s/e\~/\x{0119}/g;

  $in_str =~ s/z0/\x{017C}/g;

  $in_str =~ s/c1/\x{0107}/g;
  $in_str =~ s/l1/\x{0142}/g;
  $in_str =~ s/n1/\x{0144}/g;
  $in_str =~ s/o1/\x{00F3}/g;
  $in_str =~ s/s1/\x{015B}/g;
  $in_str =~ s/z1/\x{017A}/g;

  $in_str =~ s/o2/\x{00F6}/g;
  $in_str =~ s/u2/\x{00FC}/g;
  $in_str =~ s/c2/\x{00E7}/g;

  return NFC($in_str);  # recompose & reorder canonically
}
