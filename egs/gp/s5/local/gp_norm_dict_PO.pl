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


# This script normalizes the GlobalPhone Portuguese dictionary. It (optionally) 
# tags the phones with language ('PO') marker. It also converts the words to 
# UTF8 and lowercases everything, either of which can be diabled with command 
# line switches.
# *No special treatment for acronyms since some words are already capitalized.

my $usage = "Usage: gp_norm_dict_PO.pl [-l|-m map|-r|-u] -i dictionary > formatted\
Normalizes pronunciation dictionary for GlobalPhone Portuguese.\
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
  s/\r//g;  # Since files may have CRLF line-breaks!
  $_ =~ m:^(\S+)\s+(.+)$: or die "Bad line: $_";
  my $word = $1;
  my $pron = $2;
  next if ($pron =~ /SIL/);  # Silence will be added later to the lexicon

  # First, normalize the pronunciation:
  $pron =~ s/^\s*//; $pron =~ s/\s*$//;  # remove leading or trailing spaces
  $pron =~ s/\s+/ /g;  # Normalize spaces
  $pron = lc($pron);   # Phones in Portuguese dictionary are in uppercase

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

  $pron =~ s/(\S+)/$1_PO/g if(defined($lang_tag));

  # Next, normalize the word:
  $word =~ s/\(.*\)//g;  # Pron variants should have same orthography
  $word = &rmn2utf8_PO($word) unless (defined($keep_rmn));
  if (defined($uppercase)) {
    $word = uc($word);
  } else {
    $word = lc($word);
  }

  print "$word\t$pron\n";
}


sub rmn2utf8_PO {
  my ($in_str) = "@_";
  
  $in_str =~ s/A\:/\x{00C0}/g;
  $in_str =~ s/A\+/\x{00C1}/g;
  $in_str =~ s/A\^/\x{00C2}/g;
  $in_str =~ s/A\~/\x{00C3}/g;
  $in_str =~ s/C\:/\x{00C7}/g;
  $in_str =~ s/E\+/\x{00C9}/g;
  $in_str =~ s/E\^/\x{00CA}/g;
  $in_str =~ s/I\+/\x{00CD}/g;
  $in_str =~ s/N\~/\x{00D1}/g;
  $in_str =~ s/O\+/\x{00D3}/g;
  $in_str =~ s/O\^/\x{00D4}/g;
  $in_str =~ s/O\~/\x{00D5}/g;
  $in_str =~ s/U\+/\x{00DA}/g;
  $in_str =~ s/U\^/\x{00DC}/g;

  $in_str =~ s/a\:/\x{00E0}/g;
  $in_str =~ s/a\+/\x{00E1}/g;
  $in_str =~ s/a\^/\x{00E2}/g;
  $in_str =~ s/a\~/\x{00E3}/g;
  $in_str =~ s/c\:/\x{00E7}/g;
  $in_str =~ s/e\+/\x{00E9}/g;
  $in_str =~ s/e\^/\x{00EA}/g;
  $in_str =~ s/i\+/\x{00ED}/g;
  $in_str =~ s/n\~/\x{00F1}/g;
  $in_str =~ s/o\+/\x{00F3}/g;
  $in_str =~ s/o\^/\x{00F4}/g;
  $in_str =~ s/o\~/\x{00F5}/g;
  $in_str =~ s/u\+/\x{00FA}/g;
  $in_str =~ s/u\^/\x{00FC}/g;

  return NFC($in_str);  # recompose & reorder canonically
}
