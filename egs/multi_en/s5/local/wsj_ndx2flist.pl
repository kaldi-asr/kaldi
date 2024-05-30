#!/usr/bin/env perl

###########################################################################################
# This script was copied from egs/wsj/s5/local/ndx2flist.pl
# The source commit was e69198c3dc5633f98eb88e1cdf20b2521a598f21
# No changes were made
###########################################################################################

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

#and as command-line arguments it takes the names of the WSJ disk locations, e.g.:
#/mnt/matylda2/data/WSJ0/11-1.1 /mnt/matylda2/data/WSJ0/11-10.1  ... etc.
# It outputs a list of absolute pathnames (it does this by replacing e.g. 11_1_1 with
# /mnt/matylda2/data/WSJ0/11-1.1.
# It also does a slight fix because one of the WSJ disks (WSJ1/13-16.1) was distributed with
# uppercase rather than lower case filenames.

foreach $fn (@ARGV) {
    $fn =~ m:.+/([0-9\.\-]+)/?$: || die "Bad command-line argument $fn\n";
    $disk_id=$1; 
    $disk_id =~ tr/-\./__/; # replace - and . with - so 11-10.1 becomes 11_10_1
    $fn =~ s:/$::; # Remove final slash, just in case it is present.
    $disk2fn{$disk_id} = $fn;
}

while(<STDIN>){
    if(m/^;/){ next; } # Comment.  Ignore it.
    else {
      m/^([0-9_]+):\s*(\S+)$/  || die "Could not parse line $_";
      $disk=$1;
      if(!defined $disk2fn{$disk}) {
          die "Disk id $disk not found";
      }
      $filename = $2; # as a subdirectory of the distributed disk.
      if($disk eq "13_16_1" && `hostname` =~ m/fit.vutbr.cz/) {
          # The disk 13-16.1 has been uppercased for some reason, on the
          # BUT system.  This is a fix specifically for that case.
          $filename =~ tr/a-z/A-Z/; # This disk contains all uppercase filenames.  Why?
      }
      print "$disk2fn{$disk}/$filename\n";
  }
}
