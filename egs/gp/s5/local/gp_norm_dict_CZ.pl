#!/usr/bin/perl -w

# Copyright 2012  Arnab Ghoshal;  Milos Janda

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


# This script normalizes the GlobalPhone Czech dictionary. It (optionally) 
# tags the phones with language ('CZ') marker. It also converts the words to 
# UTF8 and lowercases everything, either of which can be diabled with command 
# line switches.
# No special treatment for acronyms since there is no easy way of detecting 
# acronyms in the dictionary.

my $usage = "Usage: gp_norm_dict_CZ.pl [-l|-u] -i dictionary > formatted\
Normalizes pronunciation dictionary for GlobalPhone Czech.\
There will probably be duplicates; so pipe the output through sort -u \
Options:\
  -l\t\tAdd language tag to the phones
  -u\t\tConvert words to uppercase (by default make everything lowercase)\n";

use strict;
use Getopt::Long;
use Unicode::Normalize;
use open ':encoding(iso-8859-2)';
binmode(STDOUT, ":encoding(utf8)");

die "$usage" unless(@ARGV >= 1);
my ($in_dict, $lang_tag, $uppercase);
GetOptions ("l"   => \$lang_tag,    # tag phones with language ID.
	    "u"   => \$uppercase,   # convert words to uppercase
            "i=s" => \$in_dict);    # Input lexicon

open(L, "<$in_dict") or die "Cannot open dictionary file '$in_dict': $!";
while (<L>) {
  s/\r//g;  # Since files may have CRLF line-breaks!
  chomp;
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
  $pron =~ s/(\S+)/$1_CZ/g if(defined($lang_tag));

  # Next, normalize the word:
  $word =~ s/\(.*\)//g;  # Pron variants should have same orthography
  if (defined($uppercase)) {
    $word = uc($word);
  } else {
    $word = lc($word);
  }

  print "$word\t$pron\n";
}
close(L);
