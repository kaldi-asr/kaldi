#!/usr/bin/perl -w

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


# This script normalizes the GlobalPhone Russian dictionary. It removes the 
# braces that separate word & pronunciation; removes the 'M_' marker from each
# phone; and (optionally) tags the phones with language ('RU') marker. It also 
# converts the words to UTF8 and lowercases everything, either of which can be 
# diabled with command line switches.
# *No special treatment for acronyms since some words are already capitalized.

my $usage = "Usage: gp_norm_dict_RU.pl [-l|-m map|-r|-u] -i dictionary > formatted\
Normalizes pronunciation dictionary for GlobalPhone Spanish.\
There will probably be duplicates; so pipe the output through sort -u \
Options:\
  -l\tAdd language tag to the phones
  -m FILE\tMapping to a different phoneset
  -r\tKeep words in GlobalPhone-style ASCII (convert to UTF8 by default)\
  -u\tConvert words to uppercase (by default make everything lowercase)\n";

use strict;
use Getopt::Long;
use Unicode::Normalize;

die "$usage" unless(@ARGV >= 1);
my ($in_dict, $lang_tag, $map_file, $keep_rmn, $uppercase);
GetOptions ("l" => \$lang_tag,    # tag phones with language ID.
 	    "m=s" => \$map_file,  # map to a different phoneset
	    "r" => \$keep_rmn,    # keep words in GlobalPhone-style ASCII (rmn)
	    "u" => \$uppercase,   # convert words to uppercase
            "i=s" => \$in_dict);  # Input lexicon

binmode(STDOUT, ":encoding(utf8)") unless (defined($keep_rmn));

my %phone_map = ();
if (defined($map_file)) {
  warn "Language tag added (-l) while mapping to different phoneset (-m)" 
      if (defined($lang_tag));
  open(M, "<$map_file") or die "Cannot open phone mapping file '$map_file': $!";
  while (<M>) {
    next if /^\#/;  # Skip comments
    s/\r//g;  # Since files may have CRLF line-breaks!
    chomp;
    next if /^$/;   # skip empty lines
    # The mapping is assumed to be: 'from-phone' 'to-phone'
    die "Bad line: $_" unless m/^(\S+)\s+(\S+).*$/;
    die "Multiple mappings for phone $1: '$2' and '$phone_map{$1}'" 
	if (defined($phone_map{$1}));
    $phone_map{$1} = $2;
  }
}

open(L, "<$in_dict") or die "Cannot open dictionary file '$in_dict': $!";
while (<L>) {
  s/\r//g;  # Since files could be in DOS format!
  chomp;
  next if /^$/;  # Skip empty lines
  next if($_=~/\#/);  # Usually incomplete or empty prons
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
  if (defined($map_file)) {
    my (@phones) = split(' ', $pron);
    for my $i (1..$#phones) {
      if (defined($phone_map{$phones[$i]})) {
	$phones[$i] = $phone_map{$phones[$i]};
      } else {
	warn "No mapping found for $phones[$i]: keeping original.";
      }
    }
    $pron = join(' ', @phones);
  }

  $pron =~ s/(\S+)/$1_RU/g if(defined($lang_tag));

  # Next, normalize the word:
  next if ($word =~ /^\$|^$|^\(|^\)/);
  $word =~ s/\(.*\)//g;  # Pron variants should have same orthography
  $word = &rmn2utf8_RU($word) unless (defined($keep_rmn));
  if (defined($uppercase)) {
    $word = uc($word);
  } else {
    $word = lc($word);
  }

  print "$word\t$pron\n";
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
