#!/usr/bin/env perl

###########################################################################################
# This script was copied from egs/hub4_english/s5/local/normalize_bn96_transcripts.pl
# The source commit was 3c96a5fdfc31408fdc0128619a5e3ee4f2cfea6f
# No change was made
###########################################################################################

# Copyright 2017  Vimal Manohar
# Apache 2.0

@ARGV == 2 ||  die "usage: hub4_normalize_bn96_transcripts.pl noise_word spoken_noise_word < transcript > transcript2";
$noise_word = shift @ARGV;
$spoken_noise_word = shift @ARGV;

while(<STDIN>) {
    $_ =~ m:^(\S+) (.+): || die "bad line $_";
    $utt = $1;
    $trans = $2;
    print "$utt";

    $trans =~ s:\(\(([^)]*)\)\):$1 :g;   # Remove unclear speech markings
    $trans =~ s:#: :g; # Remove overlapped speech markings
    $trans =~ s:\*\*([^*]+)\*\*:$1 :g;       # Remove invented word markings
    $trans =~ s:\[[^]]+\]:$noise_word :g; 
    $trans =~ s:\{[^}]+\}:$spoken_noise_word :g;
    $trans =~ s:^[+]([^+]+)[+]$:$1:;   # Remove mispronunciation brackets
    foreach $w (split (" ",$trans)) {
        $w =~ s:^@(.*)$:$1:;  # Remove best guess marking for proper nouns
        print " $w";
    }
    print "\n";
}
