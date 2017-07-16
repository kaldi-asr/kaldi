#!/usr/bin/env perl

# Copyright 2015  University of Sheffield (Author: Ning Ma)
# Apache 2.0.
#
# Computes keyword recognition accuracy (letter+digit) for the CHiME/GRID
# corpus from a transcription file containing:
#   s01_bgaa9a_0dB  BIN GREEN IN R NINE AGAIN  
#   s01_bgaa9a_3dB  BIN GREEN AT A NINE AGAIN  
#
# Usage: compute_chime1_scores.pl exp/tri1/decode_devel/scoring/trans.txt 
#

use strict;
use warnings;

my $in_list = $ARGV[0];

open my $info, $in_list or die "could not open $in_list: $!";

my %snr_scores_letter = ();
my %snr_scores_digit = ();
my %snr_count = ();
while (my $line = <$info>) {
  chomp($line);
  my @words = split /\s+/, $line;
  my @tokens = split "_", $words[0];
  my $ref = $tokens[1];
  my $snr = $tokens[2];

  # Extract letter and digit
  my $letter = uc(substr($ref, 3, 1));
  my $digit = substr($ref, 4, 1);
  if ($digit eq "z") { $digit = "ZERO" }
  elsif ($digit eq "1") { $digit = "ONE" }
  elsif ($digit eq "2") { $digit = "TWO" }
  elsif ($digit eq "3") { $digit = "THREE" }
  elsif ($digit eq "4") { $digit = "FOUR" }
  elsif ($digit eq "5") { $digit = "FIVE" }
  elsif ($digit eq "6") { $digit = "SIX" }
  elsif ($digit eq "7") { $digit = "SEVEN" }
  elsif ($digit eq "8") { $digit = "EIGHT" }
  elsif ($digit eq "9") { $digit = "NINE" }

  # Compute score
  my $nwords = scalar @words;
  if (($nwords > 4) && ($letter eq $words[4])) { $snr_scores_letter{$snr}++; }
  if (($nwords > 5) && ($digit eq $words[5])) { $snr_scores_digit{$snr}++; }
  $snr_count{$snr}++;
}

# Print out keyword accuracies
printf "\nKeyword (letter+digit) recognition accuracy (%%)\n";
printf "-----------------------------------------------------------------\n";
printf "%-10s", "SNR";
my @all_snrs = ("m6dB", "m3dB", "0dB", "3dB", "6dB", "9dB");
foreach (@all_snrs) {
  my $snr = $_;
  $snr =~ s/m/-/;
  printf "%-8s", $snr;
}
printf "%-8s", "Average";
printf "\n-----------------------------------------------------------------\n";
printf "%-10s", "Overall";
my $score_avg = 0;
my $nsnrs = scalar @all_snrs;
foreach (@all_snrs) {
  my $score = ($snr_scores_letter{$_}+$snr_scores_digit{$_})/2/$snr_count{$_}*100;
  $score_avg += $score;
  printf "%-8.2f", $score;
}
printf "%-8.2f", $score_avg/$nsnrs;
printf "\n-----------------------------------------------------------------\n";
printf "%-10s", "Letter";
$score_avg = 0;
foreach (@all_snrs) {
  my $score = $snr_scores_letter{$_}/$snr_count{$_}*100;
  $score_avg += $score;
  printf "%-8.2f", $score;
}
printf "%-8.2f", $score_avg/$nsnrs;
printf "\n";
printf "%-10s", "Digit";
$score_avg = 0;
foreach (@all_snrs) {
  my $score = $snr_scores_digit{$_}/$snr_count{$_}*100;
  $score_avg += $score;
  printf "%-8.2f", $score;
}
printf "%-8.2f", $score_avg/$nsnrs;
printf "\n-----------------------------------------------------------------\n";

