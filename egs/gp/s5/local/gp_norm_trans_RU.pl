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


# This script normalizes the GlobalPhone Russian transcripts that have been 
# extracted in a format where each line contains an utterance ID followed by
# the transcript, e.g:
# The normalization is similar to that in 'gp_format_dict_GE.pl' script.

my $usage = "Usage: gp_norm_trans_SP.pl -i transcript > formatted\
Normalizes transcriptions for GlobalPhone Spanish. The input format is \
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
  s/\r//g;  # Since files could be in DOS format!
  chomp;
  $_ =~ m:^(\S+)\s+(.+): or die "Bad line: $_";
  my $utt_id = $1;
  my $trans = $2;

  $trans =~ s/\`(.*?)\'/$1/g;  # Remove quotation marks.
  $trans =~ s/</ </g;  # Fragments are enclosed in < & > : space them properly
  $trans =~ s/>/> /g;
  
  $trans = &rmn2utf8_SP($trans) unless (defined($keep_rmn));
  if (defined($uppercase)) {
    $trans = uc($trans);
  } else {
    $trans = lc($trans);
  }

  # Normalize spaces
  $trans =~ s/^\s*//; $trans =~ s/\s*$//; $trans =~ s/\s+/ /g;
  print "$utt_id $trans\n";
}


sub rmn2utf8_RU {
  my ($in_str) = "@_";
  
  $in_str =~ s/~/\x{00D8}/g; # mekky znak - zmekcuje souhlasku pred nim napr t~ => Tb

  $in_str =~ s/schTsch/\x{00DD}/g;
  $in_str =~ s/SchTsch/\x{00DD}/g;

  $in_str =~ s/jscH/\x{00D6}/g;
  $in_str =~ s/JscH/\x{00D6}/g;
  $in_str =~ s/tscH/\x{00DE}/g;
  $in_str =~ s/TscH/\x{00DE}/g;
  $in_str =~ s/sch/\x{00DB}/g;
  $in_str =~ s/Sch/\x{00DB}/g;
  $in_str =~ s/ts/\x{00C3}/g;
  $in_str =~ s/tS/\x{00C3}/g;
  $in_str =~ s/Ts/\x{00C3}/g;
  $in_str =~ s/TS/\x{00C3}/g;

  $in_str =~ s/ye/\x{00C5}/g;
  $in_str =~ s/yo/\x{00A3}/g; # neni v textu
  $in_str =~ s/yu/\x{00C0}/g;
  $in_str =~ s/ya/\x{00D1}/g;

  $in_str =~ s/Ye/\x{00C5}/g;
  $in_str =~ s/Yo/\x{00A3}/g; # neni v textu
  $in_str =~ s/Yu/\x{00C0}/g;
  $in_str =~ s/Ya/\x{00D1}/g;

  $in_str =~ s/i2/\x{00D9}/g;
  $in_str =~ s/I2/\x{00D9}/g;
  $in_str =~ s/Q/\x{00DF}/g;

  $in_str =~ s/a/\x{00C1}/g;
  $in_str =~ s/b/\x{00C2}/g;
  $in_str =~ s/w/\x{00D7}/g;
  $in_str =~ s/g/\x{00C7}/g;
  $in_str =~ s/d/\x{00C4}/g;
  $in_str =~ s/z/\x{00DA}/g;
  $in_str =~ s/i/\x{00C9}/g;
  $in_str =~ s/j/\x{00CA}/g;
  $in_str =~ s/k/\x{00CB}/g;
  $in_str =~ s/l/\x{00CC}/g;
  $in_str =~ s/m/\x{00CD}/g;
  $in_str =~ s/n/\x{00CE}/g;
  $in_str =~ s/o/\x{00CF}/g;
  $in_str =~ s/p/\x{00D0}/g;
  $in_str =~ s/r/\x{00D2}/g;
  $in_str =~ s/s/\x{00D3}/g;
  $in_str =~ s/t/\x{00D4}/g;
  $in_str =~ s/u/\x{00D5}/g;
  $in_str =~ s/f/\x{00C6}/g;
  $in_str =~ s/h/\x{00C8}/g;
  $in_str =~ s/e/\x{00DC}/g;

  $in_str =~ s/A/\x{00C1}/g;
  $in_str =~ s/B/\x{00C2}/g;
  $in_str =~ s/W/\x{00D7}/g;
  $in_str =~ s/G/\x{00C7}/g;
  $in_str =~ s/D/\x{00C4}/g;
  $in_str =~ s/Z/\x{00DA}/g;
  $in_str =~ s/I/\x{00C9}/g;
  $in_str =~ s/J/\x{00CA}/g;
  $in_str =~ s/K/\x{00CB}/g;
  $in_str =~ s/L/\x{00CC}/g;
  $in_str =~ s/M/\x{00CD}/g;
  $in_str =~ s/N/\x{00CE}/g;
  $in_str =~ s/O/\x{00CF}/g;
  $in_str =~ s/P/\x{00D0}/g;
  $in_str =~ s/R/\x{00D2}/g;
  $in_str =~ s/S/\x{00D3}/g;
  $in_str =~ s/T/\x{00D4}/g;
  $in_str =~ s/U/\x{00D5}/g;
  $in_str =~ s/F/\x{00C6}/g;
  $in_str =~ s/H/\x{00C8}/g;
  $in_str =~ s/E/\x{00DC}/g;


  return NFC($in_str);  # recompose & reorder canonically
}
