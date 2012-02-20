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


# This script normalizes the GlobalPhone Swedish dictionary. It removes the 
# braces that separate word & pronunciation; removes the 'M_' marker from each
# phone; capitalizes the words; (optionally) tags the phones with language 
# ('SW') marker; and (optionally) puts word-begin and -end markers or 
# word-boundary markers on phones.

my $usage = "Usage: gp_format_dict_SW.pl [-l|-p|-w] -i dictionary > formatted\
Normalizes pronunciation dictionary for GlobalPhone Swedish.\
(There will probably be duplicates; so pipe the output through sort -u)\
Options:\
  -l\tAdd language tag to the phones
  -p\tUse position-dependent phones with word begin & end markings (ignores -w)\
  -w\tUse word-boundary markings on the phones\n";

use strict;
use Getopt::Long;
die "$usage" unless(@ARGV >= 1);
my ($pos_dep, $word_bound, $in_dict, $lang_tag);
GetOptions ("p" => \$pos_dep,     # position-dependent phones
            "w" => \$word_bound,  # phones with word boundary markings
            "l" => \$lang_tag,    # tag phones with language ID.
            "i=s" => \$in_dict);  # Input lexicon

open(L, "<$in_dict") or die "Cannot open dictionary file '$in_dict': $!";
while (<L>) {
  s/\r//g;  # Since files could be in DOS format!
  $_ =~ m:^\{(\S*)\}\s+\{(.+)\}\s*$: or die "Bad line: '$_'";
  my $word = $1;
  my $pron = $2;
  next if ($pron =~ /SIL/);  # Silence will be added later to the lexicon

  # First, normalize the pronunciation:
  $pron =~ s/\{//g;
  $pron =~ s/^\s*//; $pron =~ s/\s*$//;  # remove leading or trailing spaces
  if (defined($pos_dep)) {
    $pron =~ s/ WB\}$/_E/;
    $pron =~ s/ WB\}/_B/;
    $pron =~ s/_E$/_S/ unless ($pron =~ /\s+/);  # No space => singleton
  } elsif (defined($word_bound)) {
    $pron =~ s/ WB\}/_WB/g;
  } else {
    $pron =~ s/ WB\}//g;    
  }
  $pron =~ s/\s+/ /g;  # Normalize spaces
  $pron =~ s/M_//g;    # Get rid of the M_ marker before the phones
  $pron =~ s/(\S+)/$1_SW/g if(defined($lang_tag));

  # Next, normalize the word:
  next if ($word =~ /^\$|^$|^\(|^\)/);
  $word =~ s/\(.*\)//g;  # Pron variants should have same orthography
  $word =~ tr/a-z/A-Z/;  # Capitalize every word.

  print "$word\t$pron\n";
}
