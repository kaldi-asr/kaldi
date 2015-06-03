#!/usr/bin/env perl
use warnings; #sed replacement for -w perl parameter

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


# This script normalizes the GlobalPhone French transcripts that have been 
# extracted in a format where each line contains an utterance ID followed by
# the transcript, e.g:
# FR001_16 C'est un langage oriente+ objet ide+al pour les re+seaux ses programmes
# The normalization is similar to that in 'gp_norm_dict_FR.pl' script.

my $usage = "Usage: gp_norm_trans_FR.pl [-r|-u] -i transcript > formatted\
Normalizes transcriptions for GlobalPhone French. The input format is \
assumed to be utterance ID followed by transcript on the same line.
Input is assumed to be ISO-8859-1 encoded, and output is in UTF-8. \
Transcript is lowercased by default, but can be uppercased with the -u option.
\n";

use strict;
use Getopt::Long;
use Unicode::Normalize;
use open ':encoding(iso-8859-1)';
binmode(STDOUT, ":encoding(utf8)");

die "$usage" unless(@ARGV >= 1);
my ($in_trans, $keep_rmn, $uppercase);
GetOptions ("u" => \$uppercase,    # convert words to uppercase
            "r" => \$keep_rmn,     # keep words in GlobalPhone-style ASCII (rmn)
	    "i=s" => \$in_trans);  # Input transcription

open(T, "<$in_trans") or die "Cannot open transcription file '$in_trans': $!";
while (<T>) {
  s/\r//g;  # Since files may have CRLF line-breaks!
  chomp;
  $_ =~ m:^(\S+)\s+(.+): or die "Bad line: $_";
  my $utt_id = $1;
  my $trans = $2;

  $trans =~ s/\"/ /g;  # Remove quotation marks.
  $trans =~ s/[\,\.\?\!\:]/ /g;
  # Normalize spaces
  $trans =~ s/^\s*//; $trans =~ s/\s*$//; $trans =~ s/\s+/ /g;

  $trans = &rmn2utf8_FR($trans) unless (defined($keep_rmn));
  if (defined($uppercase)) {
    $trans = uc($trans);
  } else {
    $trans = lc($trans);
  }

  print "$utt_id $trans\n";
}


sub rmn2utf8_FR {
  my ($in_str) = "@_";

  # CAPITAL LETTERS
  $in_str =~ s/A\`/\x{00C0}/g; # LETTERS WITH GRAVE
  $in_str =~ s/E\`/\x{00C8}/g; 
  $in_str =~ s/U\`/\x{00D9}/g;

  $in_str =~ s/A\^/\x{00C2}/g; # LETTERS WITH CIRCUMFLEX
  $in_str =~ s/E\^/\x{00CA}/g;
  $in_str =~ s/I\^/\x{00CE}/g;
  $in_str =~ s/O\^/\x{00D4}/g;
  $in_str =~ s/U\^/\x{00DB}/g;

  $in_str =~ s/E\:/\x{00C8}/g; # LETTERS WITH DIAERESIS
  $in_str =~ s/I\:/\x{00CF}/g;
  $in_str =~ s/U\:/\x{00DC}/g;
  $in_str =~ s/Y\:/\x{0178}/g;

  $in_str =~ s/E\+/\x{00C9}/g; # E WITH ACUTE
  $in_str =~ s/C\~/\x{00C7}/g; # C WITH CEDILLA


  # SMALL LETTERS
  $in_str =~ s/a\`/\x{00E0}/g; # LETTERS WITH GRAVE
  $in_str =~ s/e\`/\x{00E8}/g; 
  $in_str =~ s/u\`/\x{00F9}/g;

  $in_str =~ s/a\^/\x{00E2}/g; # LETTERS WITH CIRCUMFLEX
  $in_str =~ s/e\^/\x{00EA}/g;
  $in_str =~ s/i\^/\x{00EE}/g;
  $in_str =~ s/o\^/\x{00F4}/g;
  $in_str =~ s/u\^/\x{00FB}/g;

  $in_str =~ s/e\:/\x{00E8}/g; # LETTERS WITH DIAERESIS
  $in_str =~ s/i\:/\x{00EF}/g; 
  $in_str =~ s/u\:/\x{00FC}/g;
  $in_str =~ s/y\:/\x{00FF}/g;

  $in_str =~ s/e\+/\x{00E9}/g; # E WITH ACUTE
  $in_str =~ s/c\~/\x{00E7}/g; # C WITH CEDILLA

  return NFC($in_str);  # recompose & reorder canonically
}
