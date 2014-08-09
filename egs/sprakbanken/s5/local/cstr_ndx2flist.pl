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

# This is modified from the script in standard Kaldi recipe to account
# for the way the WSJ data is structured on the Edinburgh systems. 
# - Arnab Ghoshal, 12/1/12

# This program takes as its standard input an .ndx file from the WSJ corpus that looks
# like this:
#;; File: tr_s_wv1.ndx, updated 04/26/94
#;;
#;; Index for WSJ0 SI-short Sennheiser training data
#;; Data is read WSJ sentences, Sennheiser mic.
#;; Contains 84 speakers X (~100 utts per speaker MIT/SRI and ~50 utts 
#;; per speaker TI) = 7236 utts
#;;
#11_1_1:wsj0/si_tr_s/01i/01ic0201.wv1
#11_1_1:wsj0/si_tr_s/01i/01ic0202.wv1
#11_1_1:wsj0/si_tr_s/01i/01ic0203.wv1

# and as command-line argument it takes the names of the WSJ disk locations, e.g.:
# /group/corpora/public/wsjcam0/data on DICE machines.
# It outputs a list of absolute pathnames.

$wsj_dir = $ARGV[0];

while(<STDIN>){
  if(m/^;/){ next; } # Comment.  Ignore it.
  else {
    m/^([0-9_]+):\s*(\S+)$/  || die "Could not parse line $_";
    $filename = $2; # as a subdirectory of the distributed disk.
    if ($filename !~ m/\.wv1$/) { $filename .= ".wv1"; }
    $filename = "$wsj_dir/$filename";
    if (-e $filename) {
      print "$filename\n";
    } else {
      print STDERR "File $filename found in the index but not on disk\n";
    }
  }
}
