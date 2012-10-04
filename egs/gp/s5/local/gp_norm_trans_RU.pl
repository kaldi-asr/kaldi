#!/usr/bin/perl -w

# Copyright 2012  Milos Janda

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
# The normalization is similar to that in 'gp_norm_dict_RU.pl' script.

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
  
  $trans = &rmn2utf8_RU($trans) unless (defined($keep_rmn));
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

  $in_str =~ s/~/\x{044C}/g; # Cyrillic Soft Sign - soften consonant before that, e.g. t~ => Tb

  $in_str =~ s/schTsch/\x{0449}/g;
  $in_str =~ s/SchTsch/\x{0429}/g;

  $in_str =~ s/jscH/\x{0436}/g;
  $in_str =~ s/JscH/\x{0416}/g;

  $in_str =~ s/tscH/\x{0447}/g;
  $in_str =~ s/TscH/\x{0427}/g;

  $in_str =~ s/sch/\x{0448}/g;
  $in_str =~ s/Sch/\x{0428}/g;

  $in_str =~ s/ts/\x{0446}/g;
  $in_str =~ s/tS/\x{0446}/g;

  $in_str =~ s/Ts/\x{0426}/g;
  $in_str =~ s/TS/\x{0426}/g;

  $in_str =~ s/ye/\x{0435}/g;
  $in_str =~ s/yo/\x{0451}/g; # non in text
  $in_str =~ s/yu/\x{044E}/g;
  $in_str =~ s/ya/\x{044F}/g;

  $in_str =~ s/Ye/\x{0415}/g;
  $in_str =~ s/Yo/\x{0401}/g; # non in text
  $in_str =~ s/Yu/\x{042E}/g;
  $in_str =~ s/Ya/\x{042F}/g;

  $in_str =~ s/i2/\x{044B}/g;
  $in_str =~ s/I2/\x{042B}/g;

  $in_str =~ s/Q/\x{044A}/g;
  $in_str =~ s/q/\x{042A}/g; # non in text

  $in_str =~ s/a/\x{0430}/g;
  $in_str =~ s/b/\x{0431}/g;
  $in_str =~ s/w/\x{0432}/g;
  $in_str =~ s/g/\x{0433}/g;
  $in_str =~ s/d/\x{0434}/g;
  $in_str =~ s/z/\x{0437}/g;
  $in_str =~ s/i/\x{0438}/g;
  $in_str =~ s/j/\x{0439}/g;
  $in_str =~ s/k/\x{043A}/g;
  $in_str =~ s/l/\x{043B}/g;
  $in_str =~ s/m/\x{043C}/g;
  $in_str =~ s/n/\x{043D}/g;
  $in_str =~ s/o/\x{043E}/g;
  $in_str =~ s/p/\x{043F}/g;
  $in_str =~ s/r/\x{0440}/g;
  $in_str =~ s/s/\x{0441}/g;
  $in_str =~ s/t/\x{0442}/g;
  $in_str =~ s/u/\x{0443}/g;
  $in_str =~ s/f/\x{0444}/g;
  $in_str =~ s/h/\x{0445}/g;
  $in_str =~ s/e/\x{044D}/g;

  $in_str =~ s/A/\x{0410}/g;
  $in_str =~ s/B/\x{0411}/g;
  $in_str =~ s/W/\x{0412}/g;
  $in_str =~ s/G/\x{0413}/g;
  $in_str =~ s/D/\x{0414}/g;
  $in_str =~ s/Z/\x{0417}/g;
  $in_str =~ s/I/\x{0418}/g;
  $in_str =~ s/J/\x{0419}/g;
  $in_str =~ s/K/\x{041A}/g;
  $in_str =~ s/L/\x{041B}/g;
  $in_str =~ s/M/\x{041C}/g;
  $in_str =~ s/N/\x{041D}/g;
  $in_str =~ s/O/\x{041E}/g;
  $in_str =~ s/P/\x{041F}/g;
  $in_str =~ s/R/\x{0420}/g;
  $in_str =~ s/S/\x{0421}/g;
  $in_str =~ s/T/\x{0422}/g;
  $in_str =~ s/U/\x{0423}/g;
  $in_str =~ s/F/\x{0424}/g;
  $in_str =~ s/H/\x{0425}/g;
  $in_str =~ s/E/\x{042D}/g;

  return NFC($in_str);  # recompose & reorder canonically
}
