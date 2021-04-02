#!/usr/bin/env perl
#===============================================================================
# Copyright 2017  (Author: Yenda Trmal <jtrmal@gmail.com>)
#
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
#===============================================================================

use strict;
use warnings;
use utf8;

binmode STDIN, ":utf8";
binmode STDOUT, ":utf8";
binmode STDERR, ":utf8";

my $lexicon_name = $ARGV[0];
open(my $lexicon_file, "<:encoding(UTF-8)", $lexicon_name) or
  die "Cannot open $lexicon_name: $!\n";

my $wordlist_name = $ARGV[1];
open(my $wordlist_file, "<:encoding(UTF-8)", $wordlist_name) or
  die "Cannot open $wordlist_name: $!\n";


my %lexicon;
while (<$lexicon_file>) {
  chomp;
  (my $word, my $prons) = split " ", $_, 2;
  $lexicon{uc $word} = $prons;
}

while (<$wordlist_file>) {
  chomp;
  my $word = $_;
  print STDERR "Cannot find word $word in lexicon\n" unless defined($lexicon{uc $word});

  #print "$word $lexicon{$word}\n";

  my @prons = split "\t", $lexicon{uc $word};
  foreach my $pron (@prons) {
    my @phones = split " ", $pron;
    my $stress_mark = 0;
    my @out_phones = ();
    foreach my $phone (@phones) {
      if ($phone eq "\"") {
        $stress_mark = 1
      } elsif ( $phone eq "." ) {
        $stress_mark = 0;
        push @out_phones, '.';
      } elsif ( $phone eq "#" ) {
        $stress_mark = 0;
        push @out_phones, '.';
      } else {
        $phone =~ s/_/+/g;
        #let's just ignore stress for now
        #$phone = "${phone}_\"" if $stress_mark;
        push @out_phones, $phone;
      }
    }
    my $out_pron = join(" ", @out_phones);
    $out_pron =~ s/ *\. */\t/g;
    print "$word\t$out_pron\n";
  }
}

