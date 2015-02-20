#!/usr/bin/perl
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

@ARGV == 1 || die "find_transcripts.pl dot_files_flist < utterance_ids > transcripts";
$dot_flist = shift @ARGV;

open(L, "<$dot_flist") || die "Opening file list of dot files: $dot_flist\n";
while(<L>){
    chop;
    m:\S+/(\w{6})00.dot: || die "Bad line in dot file list: $_";
    $spk = $1;
    $spk2dot{$spk} = $_;
}



while(<STDIN>){ 
    chop;
    $uttid_orig = $_;
    $uttid = substr $uttid_orig, 0, 8; 
    $uttid =~ m:(\w{6})\w\w: || die "Bad utterance id $_";
    $spk = $1;
    if($spk ne $curspk) {
        %utt2trans = { }; # Don't keep all the transcripts in memory...
        $curspk = $spk;
        $dotfile = $spk2dot{$spk};
        defined $dotfile || die "No dot file for speaker $spk\n";
        open(F, "<$dotfile") || die "Error opening dot file $dotfile\n";
        while(<F>) {
            $_ =~ m:(.+)\((\w{8})\)\s*$: || die "Bad line $_ in dot file $dotfile (line $.)\n";
            $trans = $1;
            $utt = $2;
            $utt2trans{$utt} = $trans;
        }
    }
    if(!defined $utt2trans{$uttid}) {
        print STDERR "No transcript for utterance $uttid (current dot file is $dotfile)\n";
    } else {
        print "$uttid_orig $utt2trans{$uttid}\n";
    }
}


