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
# phone; capitalizes the words; (optionally) puts '-' between the letters of
# acronyms; (optionally) tags the phones with language ('GE') marker; and 
# (optionally) puts word-begin and -end markers or word-boundary markers on 
# phones.

my $usage = "Usage: gp_format_dict_GE.pl [-a|-l|-p|-w] -i dictionary > formatted\
Normalizes pronunciation dictionary for GlobalPhone German.\
(There will probably be duplicates; so pipe the output through sort -u)\
Options:\
  -a\tTreat acronyms differently (puts - between individual letters)
  -l\tAdd language tag to the phones
  -p\tUse position-dependent phones with word begin & end markings (ignores -w)\
  -w\tUse word-boundary markings on the phones\n";

use strict;
use Getopt::Long;
die "$usage" unless(@ARGV >= 1);
my ($acro, $pos_dep, $word_bound, $in_dict, $lang_tag);
GetOptions ("p" => \$pos_dep,     # position-dependent phones
            "w" => \$word_bound,  # phones with word boundary markings
            "a" => \$acro,        # put - between letters of acronyms
            "l" => \$lang_tag,    # tag phones with language ID.
            "i=s" => \$in_dict);  # Input lexicon

open(L, "<$in_dict") or die "Cannot open dictionary file '$in_dict': $!";
while (<L>) {
  s/\r//g;  # Since files could be in DOS format!
  next if($_=~/\+|\=|^\{\'|^\{\-|\<_T\>/);  # Usually incomplete or empty prons
  $_ =~ m:^\{(\S+)\}\s+\{(.+)\}$: or die "Bad line: $_";
  my $word = $1;
  my $pron = $2;
  next if ($pron =~ /SIL/);  # Silence will be added later to the lexicon

  # First, normalize the pronunciation:
  $pron =~ s/\{//g;
  $pron =~ s/^\s*//; $pron =~ s/\s*$//;  # remove leading or trailing spaces
  if (defined($pos_dep)) {
    $pron =~ s/ WB\}$/_E/;
    $pron =~ s/ WB\} /_B /;
    $pron =~ s/_E$/_S/ unless ($pron =~ /\s+/);  # No space => singleton
  } elsif (defined($word_bound)) {
    $pron =~ s/ WB\}/_WB/g;
  } else {
    $pron =~ s/ WB\}//g;    
  }
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

  # Distinguish acronyms before capitalizing everything, since they have 
  # different pronunciations. This may not be important.
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

  $word =~ tr/a-z/A-Z/;  # Now, capitalize every word.

  print "$word\t$pron\n";
}
