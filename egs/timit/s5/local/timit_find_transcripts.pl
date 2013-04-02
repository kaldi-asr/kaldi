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
# Extracts from the phn files the transcripts for a given
# dataset (represented by a file list).
# 

@ARGV == 1 || die "timit_find_transcripts.pl phn_trans_flist < utterance_ids > transcripts";
$phn_flist = shift @ARGV;

open(L, "<$phn_flist") || die "Opening file list of phn files: $phn_flist\n";
while(<L>){
    chop;
    m:^\S+/(\w+)/(\w+)\.[pP][hH][nN]$: || die "Bad line in phn file list: $_";
    $spk = $1 . "_" . $2;
    $spk2phn{$spk} = $_;
}

%utt2trans = { }; 
while(<STDIN>){ 
    chop;
    $uttid = $_;
    $uttid =~ m:(\w+)_(\w+): || die "Bad utterance id $_";
    $phnfile = $spk2phn{$uttid};
    defined $phnfile || die "No phn file for speaker $spk\n";
    open(F, "<$phnfile") || die "Error opening phn file $phnfile\n";
    @trans = "";
    while(<F>) {
        $_ =~ m:\d+\s\d+\s(.+)$: || die "Bad line $_ in phn file $phnfile (line $.)\n";
        push (@trans,$1);
    }
    $utt2trans{$uttid} = join(" ",@trans);        

    if(!defined $utt2trans{$uttid}) {
        print STDERR "No transcript for utterance $uttid (current phn file is $phnfile)\n";
    } else {
        print "$uttid $utt2trans{$uttid}\n";
    }
    close(F);
}


