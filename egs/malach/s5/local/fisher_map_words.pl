#!/usr/bin/env perl
use warnings; #sed replacement for -w perl parameter

# Copyright 2013  Arnab Ghoshal

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


# This script cleans up the Fisher English transcripts and maps the words to
# be similar to the Switchboard Mississippi State transcripts
# Reads from STDIN and writes to STDOUT

use strict;

while (<>) {
  chomp;

  $_ = lc($_);  # few things aren't lowercased in the data, e.g. I'm
  s/\*//g;  # *mandatory -> mandatory
  s/\(//g;  s/\)//g;  # Remove parentheses
  next if /^\s*$/;    # Skip empty lines

  # In one conversation people speak some German phrases that are tagged as
  # <german (( ja wohl )) > -- we remove these
  s/<[^>]*>//g;

  s/\.\_/ /g;  # Abbreviations: a._b._c. -> a b c.
  s/(\w)\.s( |$)/$1's /g;  # a.s -> a's
  s/\./ /g;    # Remove remaining .
  s/(\w)\,(\w| )/$1 $2/g;    # commas don't appear within numbers, but still

  s/( |^)\'(blade|cause|course|frisco|okay|plain|specially)( |$)/ $2 /g;
  s/\'em/-em/g;

  # Remove an opening ' if there is a matching closing ' since some word 
  # fragments are annotated as: 'kay, etc.
  # The substitution is done twice, since matching once doesn't capture 
  # consequetive quoted segments (the space in between is used up).
  s/(^| )\'(.*?)\'( |$)/ $2 /g;
  s/(^| )\'(.*?)\'( |$)/ $2 /g;

  s/( |^)\'(\w)( |-|$)/$1 /g;  # 'a- -> a
  s/( |^)-( |$)/ /g;      # Remove dangling -
  s/\?//g;                # Remove ?
  s/( |^)non-(\w+)( |$)/ non $2 /g;  # non-stop -> non stop

  # Some words that are annotated as fragments are actual dictionary words
  s/( |-)(acceptable|arthritis|ball|cause|comes|course|eight|eighty|field|giving|habitating|heard|hood|how|king|ninety|okay|paper|press|scripts|store|till|vascular|wood|what|york)(-| )/ $2 /g;

  # Remove [[skip]] and [pause]
  s/\[\[skip\]\]/ /g;  
  s/\[pause\]/ /g;

  # [breath], [cough], [lipsmack], [sigh], [sneeze] -> [noise]
  s/\[breath\]/[noise]/g;
  s/\[cough\]/[noise]/g;
  s/\[lipsmack\]/[noise]/g;
  s/\[sigh\]/[noise]/g;
  s/\[sneeze\]/[noise]/g;

  s/\[mn\]/[vocalized-noise]/g;  # [mn] -> [vocalized-noise]
  s/\[laugh\]/[laughter]/g;      # [laugh] -> [laughter]

  $_ = uc($_);
  # Now, mapping individual words
  my @words = split /\s+/;
  for my $i (0..$#words) {
    my $w = $words[$i];
    $w =~ s/^'/-/;
    $words[$i] = $w;
  }
  print join(" ", @words) . "\n";
}
