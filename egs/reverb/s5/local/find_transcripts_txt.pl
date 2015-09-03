#!/usr/bin/env perl
# Copyright 2013 MERL (author: Felix Weninger)
# Modified from original Kaldi code: find_transcripts.pl

# Copyright 2010-2011 Microsoft Corporation

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



# This program takes on its standard input a list of utterance
# id's, one for each line. (e.g. 4k0c030a is a an utterance id).

# This is like find_transcripts_singledot.pl but using Kaldi txt as input format.
# It also does not assume 8-character utterance IDs since this is not the case
# for MCWSJAV, for which this script is used.

use strict;

my $txtfile;
my %utt2trans;

@ARGV == 1 || die "$0 txt_file < utterance_ids > transcripts";
$txtfile = shift @ARGV;

open(F, "<$txtfile") || die "Error opening txt file $txtfile\n";
while(<F>) {
    $_ =~ m:^(\w+)\s+(.+)$: || die "Bad line $_ in txt file $txtfile (line $.)\n";
    my $trans = $2;
    my $utt = $1;
    #print "utt = $utt\n";
    $utt2trans{$utt} = $trans;
}


while(<STDIN>){ 
    chop;
    my $uttid = $_;
    $uttid =~ m:\w+: || die "Bad utterance id $_";
    if(!defined $utt2trans{$uttid}) {
        print STDERR "No transcript for utterance $uttid (current dot file is $txtfile)\n";
    } else {
        print "$uttid $utt2trans{$uttid}\n";
    }
}


