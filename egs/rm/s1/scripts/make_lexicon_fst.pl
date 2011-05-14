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


# makes lexicon FST (no pron-probs involved).

if(@ARGV != 1 && @ARGV != 3) {
    die "Usage: make_lexicon_fst.pl lexicon.txt [silprob silphone] > lexiconfst.txt"
}

$lexfn = shift @ARGV;
if(@ARGV == 0) {
    $silprob = 0.0;
} else { 
    ($silprob,$silphone) = @ARGV;
}
if($silprob != 0.0) {
    $silprob < 1.0 || die "Sil prob cannot be >= 1.0";
    $silcost = -log($silprob);
    $nosilcost = -log(1.0 - $silprob);
}


open(L, "<$lexfn") || die "Error opening lexicon $lexfn";



if( $silprob == 0.0 ) { # No optional silences: just have one (loop+final) state which is numbered zero.
    $loopstate = 0;
    $nexststate = 1; # next unallocated state.
    while(<L>) {
        @A = split(" ", $_);
        $w = shift @A;
        if(@A == 0) { # For empty words (<s> and </s>) insert no optional
                      # silence (not needed as adjacent words supply it)....
                      # actually we only hit this case for the lexicon without disambig
                      # symbols but doesn't ever matter as training transcripts don't have <s> or </s>.
            print "$loopstate\t$loopstate\t<eps>\t$w\n";
        } else {
            $s = $loopstate;
            $word_or_eps = $w;
            while (@A > 0) {
                $p = shift @A;
                if(@A > 0) {
                    $ns = $nextstate++;
                } else {
                    $ns = $loopstate;
                }
                print "$s\t$ns\t$p\t$word_or_eps\n";
                $word_or_eps = "<eps>";
                $s = $ns;
            }            
        }
    }
    print "$loopstate\t0\n"; # final-cost.
} else { # have silence probs.
    $startstate = 0;
    $loopstate = 1;
    $silstate = 2; # state from where we go to loopstate after emitting silence.
    $nextstate = 3;
    print "$startstate\t$loopstate\t<eps>\t<eps>\t$nosilcost\n"; # no silence.
    print "$startstate\t$loopstate\t$silphone\t<eps>\t$silcost\n"; # silence.
    print "$silstate\t$loopstate\t$silphone\t<eps>\n"; # no cost.
    while(<L>) {
        @A = split(" ", $_);
        $w = shift @A;
        if(@A == 0) { # For empty words (<s> and </s>) insert no optional
                      # silence (not needed as adjacent words supply it)....
                      # actually we only hit this case for the lexicon without disambig
                      # symbols but doesn't ever matter as training transcripts don't have <s> or </s>.
            print "$loopstate\t$loopstate\t<eps>\t$w\n";
        } else { 
            $is_silence_word = (@A == 1 && $A[0] eq $silphone); # boolean.
            $s = $loopstate;
            $word_or_eps = $w;
            while (@A > 0) {
                $p = shift @A;
                if(@A > 0) {
                    $ns = $nextstate++;
                    print "$s\t$ns\t$p\t$word_or_eps\n";
                    $word_or_eps = "<eps>";
                    $s = $ns;
                } else {
                    if(! $is_silence_word) {  
                        # This is non-deterministic but relatively compact,
                        # and avoids epsilons.
                        print "$s\t$loopstate\t$p\t$word_or_eps\t$nosilcost\n";
                        print "$s\t$silstate\t$p\t$word_or_eps\t$silcost\n";
                    } else {
                        # no point putting opt-sil after silence word.
                        print "$s\t$loopstate\t$p\t$word_or_eps\n";
                    }
                    $word_or_eps = "<eps>";
                }
            }
        }            
    }
    print "$loopstate\t0\n"; # final-cost.
}
