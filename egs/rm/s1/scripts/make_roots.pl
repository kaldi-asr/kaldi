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


# Written by Dan Povey 9/21/2010.  Apache 2.0 License.

# This version of make_roots.pl is specialized for RM.

# This script creates the file roots.txt which is an input to train-tree.cc.  It
# specifies how the trees are built.  The input file phone-sets.txt is a partial
# version of roots.txt in which phones are represented by their spelled form, not
# their symbol id's.  E.g. at input, phone-sets.txt might contain;
#  shared not-split  sil
# Any phones not specified in phone-sets.txt but present in phones.txt will
# be given a default treatment.  If the --separate option is given, we create
# a separate tree root for each of them, otherwise they are all lumped in one set.
# The arguments shared|not-shared and split|not-split are needed if any
# phones are not specified in phone-sets.txt.  What they mean is as follows:
# if shared=="shared" then we share the tree-root between different HMM-positions
# (0,1,2).  If split=="split" then we actually do decision tree splitting on
# that root, otherwise we forbid decision-tree splitting.  (The main reason we might 
# set this to false is for silence when
# we want to ensure that the HMM-positions will remain with a single PDF id.


$separate = 0;
if($ARGV[0] eq "--separate") {
    $separate = 1;
    shift @ARGV;
}

if(@ARGV != 4) {
    die "Usage: make_roots.pl [--separate] phones.txt silence-phone-list[integer,colon-separated] shared|not-shared split|not-split > roots.txt\n";
}


($phonesfile, $silphones, $shared, $split) = @ARGV;
if($shared ne "shared" && $shared ne "not-shared") {
    die "Third argument must be \"shared\" or \"not-shared\"\n";
}
if($split ne "split" && $split ne "not-split") {
    die "Third argument must be \"split\" or \"not-split\"\n";
}



open(F, "<$phonesfile") || die "Opening file $phonesfile";

while(<F>) {
    @A = split(" ", $_);
    if(@A != 2) {
        die "Bad line in phones symbol file: ".$_;
    }
    if($A[1] != 0) {
        $symbol2id{$A[0]} = $A[1];
        $id2symbol{$A[1]} = $A[0];
    }
}

if($silphones == ""){ 
    die "Empty silence phone list in make_roots.pl";
}
foreach $silphoneid (split(":", $silphones)) {
    defined $id2symbol{$silphoneid} || die "No such silence phone id $silphoneid";
    # Give each silence phone its own separate pdfs in each state, but
    # no sharing (in this recipe; WSJ is different.. in this recipe there
    #is only one silence phone anyway.)
    $issil{$silphoneid} = 1;
    print "not-shared not-split $silphoneid\n";
}

$idlist = "";
$remaining_phones = "";

if($separate){
    foreach $a (keys %id2symbol) {
        if(!defined $issil{$a}) {
            print "$shared $split $a\n";
        }
    }
} else {
    print "$shared $split ";
    foreach $a (keys %id2symbol) {
        if(!defined $issil{$a}) {
            print "$a ";
        }
    }
    print "\n";
}
