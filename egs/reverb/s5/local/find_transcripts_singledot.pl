#!/usr/bin/env perl
# Copyright 2013 MERL (author: Felix Weninger and Shinji Watanabe)
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
# It takes as
# Extracts from the dot files the transcripts for a given
# dataset (represented by a file list).
# 

use strict;

my $dotfile;
my %utt2trans;

@ARGV == 1 || die "find_transcripts_singledot.pl dot_file < utterance_ids > transcripts";
$dotfile = shift @ARGV;

open(F, "<$dotfile") || die "Error opening dot file $dotfile\n";
while(<F>) {
    $_ =~ m:(.+)\((\w{8})\)\s*$: || die "Bad line $_ in dot file $dotfile (line $.)\n";
    my $trans = $1;
    my $utt = $2;
    $utt2trans{$utt} = $trans;
}


while(<STDIN>){ 
    chop;
    my $uttid = $_;
    $uttid =~ m:\w{8}: || die "Bad utterance id $_";
    if(!defined $utt2trans{$uttid}) {
        print STDERR "No transcript for utterance $uttid (current dot file is $dotfile)\n";
    } else {
        print "$uttid $utt2trans{$uttid}\n";
    }
}


