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


# takes a transcript file with lines like
# 40po031e THE RATE FELL TO SIX %PERCENT IN NOVEMBER NINETEEN EIGHTY SIX .PERIOD
# on the standard input.
# The first (and only) command-line argument is the filename of a dictionary file with lines like
# ZYUGANOV  Z Y UW1 G AA0 N AA0 V
# This file replaces all OOVs with the spoken-noise word and prints counts for each OOV on the standard error.

@ARGV == 2 || die "Usage: oov2unk.pl dict spoken-noise-word < transcript > transcript2";

$dict = shift @ARGV;
open(F, "<$dict") || die "Died opening dictionary file $dict\n";
while(<F>){
   @A = split(" ", $_);
   $word = shift @A;
   $seen{$word} = 1;
}
$spoken_noise_word = shift @ARGV;

while(<STDIN>) {
   @A = split(" ", $_);
   $utt = shift @A;
   print $utt;
   foreach $a (@A) {
       if(defined $seen{$a}) {
           print " $a";
       } else  { 
           $oov{$a}++;
           print " $spoken_noise_word";
       }
   }
   print "\n";
}


foreach $w (sort { $oov{$a} <=> $oov{$b} } keys %oov) {
    print STDERR "$w $oov{$w}\n";
}
