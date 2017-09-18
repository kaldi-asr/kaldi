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

@ARGV == 1 ||  die "usage: normalize_transcript.pl noise_word < transcript > transcript2";
$noise_word = shift @ARGV;

while(<STDIN>) {
    $_ =~ m:^(\S+) (.+): || die "bad line $_";
    $utt = $1;
    $trans = $2;
    print "$utt";
    foreach $w (split (" ",$trans)) {
        $w =~ tr:a-z:A-Z:; # Upcase everything to match the CMU dictionary. .
        $w =~ s:\\::g;      # Remove backslashes.  We don't need the quoting.
        $w =~ s:^\%PERCENT$:PERCENT:; # Normalization for Nov'93 test transcripts.
        $w =~ s:^\.POINT$:POINT:; # Normalization for Nov'93 test transcripts.
        if($w =~ m:^\[\<\w+\]$:  || # E.g. [<door_slam], this means a door slammed in the preceding word. Delete.
           $w =~ m:^\[\w+\>\]$:  ||  # E.g. [door_slam>], this means a door slammed in the next word.  Delete.
           $w =~ m:\[\w+/\]$: ||  # E.g. [phone_ring/], which indicates the start of this phenomenon.
           $w =~ m:\[\/\w+]$: ||  # E.g. [/phone_ring], which indicates the end of this phenomenon.
           $w eq "~" ||        # This is used to indicate truncation of an utterance.  Not a word.
           $w eq ".") {      # "." is used to indicate a pause.  Silence is optional anyway so not much
                             # point including this in the transcript.
            next; # we won't print this word.
        } elsif($w =~ m:\[\w+\]:) { # Other noises, e.g. [loud_breath].
            print " $noise_word";
        } elsif($w =~ m:^\<([\w\']+)\>$:) {
            # e.g. replace <and> with and.  (the <> means verbal deletion of a word).. but it's pronounced.
            print " $1";
        } elsif($w eq "--DASH") {
            print " -DASH";  # This is a common issue; the CMU dictionary has it as -DASH.
#        } elsif($w =~ m:(.+)\-DASH$:) { # E.g. INCORPORATED-DASH... seems the DASH gets combined with previous word
#            print " $1 -DASH";
        } else {
            print " $w";
        }
    }
    print "\n";
}
