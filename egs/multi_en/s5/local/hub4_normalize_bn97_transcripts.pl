#!/usr/bin/env perl

###########################################################################################
# This script was copied from egs/hub4_english/s5/local/normalize_bn97_transcripts.pl
# The source commit was 148c060d8593386ee29cfcef8a2a0a050c67bce6
# No change was made
###########################################################################################

# Copyright 2017  Vimal Manohar
# Apache 2.0

@ARGV == 2 ||  die "usage: hub4_normalize_bn97_transcripts.pl noise_word spoken_noise_word < transcript > transcript2";
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
    $trans =~ s:^[+]([^+]+)[+]$:$1:;   # Remove mispronunciation brackets
    foreach $w (split (" ",$trans)) {
        if ($w ne $noise_word && $w ne $spoken_noise_word) {
          $w =~ s:[?.,!]+$::;   # Remove punctuations
          $w =~ s:^@(.*)$:$1:;  # Remove best guess marking for proper nouns
          $w =~ s:^[\^](.*)$:$1:;  # Remove capitalization marks
          $w =~ s:_([A-Z])'S$:$1.'S :g;  # Normalize abbreviations from _f_b_i to f. b. i.
          $w =~ s:_([A-Z]):$1. :g;  # Normalize abbreviations from _f_b_i to f. b. i.
          $w =~ s:[ ]+$::;  # Remove trailing spaces
        }

        print " $w";
    }
    print "\n";
}
