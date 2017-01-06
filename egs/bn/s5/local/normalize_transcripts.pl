#!/usr/bin/env perl
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


# This takes data from the standard input that's unnormalized transcripts in the format
# 4k2c0308 Of course there isn\'t any guarantee the company will keep its hot hand [misc_noise] 
# 4k2c030a [loud_breath] And new hardware such as the set of personal computers I\. B\. M\. introduced last week can lead to unexpected changes in the software business [door_slam] 
# and outputs normalized transcripts.
# c.f. /mnt/matylda2/data/WSJ0/11-10.1/wsj0/transcrp/doc/dot_spec.doc

@ARGV == 2 ||  die "usage: normalize_transcript.pl noise_word < transcript > transcript2";
$noise_word = shift @ARGV;
$spoken_noise_word = shift @ARGV;

while(<STDIN>) {
    $_ =~ m:^(\S+) (.+): || die "bad line $_";
    $utt = $1;
    $trans = $2;
    print "$utt";

    $trans =~ tr:a-z:A-Z:;
    $trans =~ s:\(\(([^)]*)\)\):$1 :g;   # Remove unclear speech markings
    $trans =~ s:#: :g; # Remove overlapped speech markings
    $trans =~ s:\*\*([^*]+)\*\*:$1 :g;       # Remove invented word markings
    $trans =~ s:\[[^]]+\]:$noise_word :g; 
    $trans =~ s:\{[^}]+\}:$spoken_noise_word :g;
    foreach $w (split (" ",$trans)) {
        $w =~ s:^[+](.+)[+]$:$1:;   # Remove mispronunciation brackets
        $w =~ s:^@(.*)$:$1:;  # Remove best guesses for proper nouns
        print " $w";
    }
    print "\n";
}

