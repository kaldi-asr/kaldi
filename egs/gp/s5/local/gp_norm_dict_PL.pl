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


# This script normalizes the GlobalPhone Polish dictionary. It (optionally) 
# tags the phones with language ('PL') marker. It also converts the words to 
# UTF8 and lowercases everything, either of which can be diabled with command 
# line switches.
# *No special treatment for acronyms since some words are already capitalized.

my $usage = "Usage: gp_norm_dict_PL.pl [-l|-m map|-u] -i dictionary > formatted\
Normalizes pronunciation dictionary for GlobalPhone Polish.\
There will probably be duplicates; so pipe the output through sort -u \
Options:\
  -l\tAdd language tag to the phones
  -m FILE\tMapping to a different phoneset
  -u\tConvert words to uppercase (by default make everything lowercase)\n";

use strict;
use Getopt::Long;
use Unicode::Normalize;
use open ':encoding(utf8)';
binmode(STDOUT, ":encoding(utf8)");

die "$usage" unless(@ARGV >= 1);
my ($in_dict, $lang_tag, $map_file, $keep_rmn, $uppercase);
GetOptions ("l" => \$lang_tag,    # tag phones with language ID.
	    "m=s" => \$map_file,  # map to a different phoneset
	    "r" => \$keep_rmn,    # keep words in GlobalPhone-style ASCII (rmn)
	    "u" => \$uppercase,   # convert words to uppercase
            "i=s" => \$in_dict);  # Input lexicon

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
  #$_ = NFD($_);  # NO UTF8 decompose
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

  $pron =~ s/(\S+)/$1_PL/g if (defined($lang_tag));

  # Next, normalize the word:
  $word =~ s/\(.*\)//g;  # Pron variants should have same orthography
  $word = &rmn2utf8_PL($word) unless (defined($keep_rmn));
  if (defined($uppercase)) {
    $word = uc($word);
  } else {
    $word = lc($word);
  }

  print "$word\t$pron\n";
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

  # OTHER STUFF
  $in_str =~ s/a1/a}/g;
  $in_str =~ s/A1/A}/g;
  $in_str =~ s/e1/e}/g;
  $in_str =~ s/E1/E}/g;

  return NFC($in_str);  # recompose & reorder canonically
}
