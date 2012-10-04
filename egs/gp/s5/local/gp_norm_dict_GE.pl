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


# This script normalizes the GlobalPhone German dictionary. It removes the 
# braces that separate word & pronunciation; removes the 'M_' marker from each
# phone; (optionally) puts '-' between the letters of acronyms; and 
# (optionally) tags the phones with language ('GE') marker. It also converts 
# the words to UTF8 and lowercases everything, either of which can be diabled 
# with command line switches.

my $usage = "Usage: gp_norm_dict_GE.pl [-a|-l|-r|-u] -i dictionary > formatted \
Normalizes pronunciation dictionary for GlobalPhone German.\
There will probably be duplicates; so pipe the output through sort -u \
Options:\
  -a\tTreat acronyms differently (puts - between individual letters)
  -l\tAdd language tag to the phones
  -r\tKeep words in GlobalPhone-style ASCII (convert to UTF8 by default)\
  -u\tConvert words to uppercase (by default make everything lowercase)\n";

use strict;
use Getopt::Long;
use Unicode::Normalize;

die "$usage" unless(@ARGV >= 1);
my ($acro, $in_dict, $lang_tag, $keep_rmn, $uppercase);
GetOptions ("a" => \$acro,        # put - between letters of acronyms
            "l" => \$lang_tag,    # tag phones with language ID.
            "r" => \$keep_rmn,    # keep words in GlobalPhone-style ASCII (rmn)
	    "u" => \$uppercase,   # convert words to uppercase
            "i=s" => \$in_dict);  # Input lexicon

binmode(STDOUT, ":encoding(utf8)") unless (defined($keep_rmn));

open(L, "<$in_dict") or die "Cannot open dictionary file '$in_dict': $!";
while (<L>) {
  s/\r//g;  # Since files may have CRLF line-breaks!
  next if($_=~/\+|\=|^\{\'|^\{\-|\<_T\>/);  # Usually incomplete or empty prons
  $_ =~ m:^\{?(\S*?)\}?\s+\{?(.+?)\}?$: or die "Bad line: $_";
  my $word = $1;
  my $pron = $2;
  next if ($pron =~ /SIL/);  # Silence will be added later to the lexicon

  # First, normalize the pronunciation:
  $pron =~ s/\{//g;
  $pron =~ s/^\s*//; $pron =~ s/\s*$//;  # remove leading or trailing spaces
  $pron =~ s/ WB\}//g;    
  $pron =~ s/\s+/ /g;  # Normalize spaces
  $pron =~ s/M_//g;    # Get rid of the M_ marker before the phones
  $pron =~ s/(\S+)/$1_GE/g if(defined($lang_tag));

  # Next, normalize the word:
  $word =~ s/\(.*\)//g;  # Pron variants should have same orthography
  $word =~ s/\$//g;      # Some letters & acronyms written with $, e.g $A
  $word =~ s/^\%//;      # Not sure what these words are, but they seem to have 
                         # correct pronunciations. So include them.
  next if($word =~ /^\'|^\-|^$|^\(|^\)|^\*/);

  # Check for spurious prons: quick & dirty!
  my @w = split(//, $word);
  my @p = split(/ /, $pron);
  next if (scalar(@p)<=5 && scalar(@w)>scalar(@p)+5);

  # Distinguish acronyms before changing case, since they have different 
  # pronunciations. This may not be important.
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

  $word = &rmn2utf8_GE($word) unless (defined($keep_rmn));
  if (defined($uppercase)) {
    $word = uc($word);
  } else {
    $word = lc($word);
  }

  print "$word\t$pron\n";
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
