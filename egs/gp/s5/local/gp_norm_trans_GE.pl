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


# This script normalizes the GlobalPhone German transcripts encoded in 
# GlobalPhone style ASCII (rmn) that have been extracted in a format where each 
# line contains an utterance ID followed by the transcript, e.g:
# GE008_10 man mag es drehen und wenden wie man will
# The normalization is similar to that in 'gp_norm_dict_GE.pl' script.

my $usage = "Usage: gp_format_trans_GE.pl [-a|-r|-u] -i transcript > formatted\
Normalizes transcriptions for GlobalPhone German. The input format is assumed\
to be utterance ID followed by transcript on the same line.\
Options:\
  -a\tTreat acronyms differently (puts - between individual letters)\
  -r\tKeep words in GlobalPhone-style ASCII (convert to UTF8 by default)\
  -u\tConvert words to uppercase (by default make everything lowercase)\n";

use strict;
use Getopt::Long;
use Unicode::Normalize;

die "$usage" unless(@ARGV >= 1);
my ($acro, $in_trans, $keep_rmn, $uppercase);
GetOptions ("a" => \$acro,         # put - between letters of acronyms
            "r" => \$keep_rmn,     # keep words in GlobalPhone-style ASCII (rmn)
	    "u" => \$uppercase,    # convert words to uppercase
            "i=s" => \$in_trans);  # Input transcription

binmode(STDOUT, ":encoding(utf8)") unless (defined($keep_rmn));

open(T, "<$in_trans") or die "Cannot open transcription file '$in_trans': $!";
while (<T>) {
  s/\r//g;  # Since files may have CRLF line-breaks!
  s/\$//g;  # Some letters & acronyms written with $, e.g $A
  chomp;
  $_ =~ m:^(\S+)\s+(.+): || die "Bad line: $_";
  my $utt_id = $1;
  my $trans = $2;

  $trans =~ s/^\s*//; $trans =~ s/\s*$//;  # Normalize spaces
  $trans =~ s/ \,(.*?)\'/ $1/g;  # Remove quotation marks.
  $trans =~ s/ \-/ /g;  # conjoined noun markers, don't need them.
  $trans =~ s/\- / /g;  # conjoined noun markers, don't need them.
  $trans = &rmn2utf8_GE($trans) unless (defined($keep_rmn));

  print $utt_id;
  for my $word (split(/\s+/, $trans)) {
    # Distinguish acronyms before capitalizing everything, since they have 
    # different pronunciations. This may not be important.
    if (defined($acro)) {
      if ($word =~ /^[\p{Lu}-]+(\-.*)*$/) {
	my @subwords = split('-', $word);
	$word = "";
	for my $i (0..$#subwords) {
	  if($subwords[$i] =~ /^[\p{Lu}]{2,}$/) {
	    $subwords[$i] = join('-', split(//, $subwords[$i]));
	  }
	}
	$word = join('-', @subwords);
      }
    }

    if (defined($uppercase)) {
      $word = uc($word);
    } else {
      $word = lc($word);
    }
    print " $word"
  }
  print "\n";
}



sub rmn2utf8_GE {
  my ($in_str) = "@_";
  
  $in_str =~ s/\~A/\x{00C4}/g;
  $in_str =~ s/\~O/\x{00D6}/g;
  $in_str =~ s/\~U/\x{00DC}/g;
  
  $in_str =~ s/\~a/\x{00E4}/g;
  $in_str =~ s/\~o/\x{00F6}/g;
  $in_str =~ s/\~u/\x{00FC}/g;
  $in_str =~ s/\~s/\x{00DF}/g;

  return NFC($in_str);  # recompose & reorder canonically
}
