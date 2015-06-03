#!/usr/bin/env perl
use warnings; #sed replacement for -w perl parameter
# Copyright 2012  Johns Hopkins University (author: Daniel Povey)

# makes lexicon FST -- special version only for use in keyword search
# for allowing optional silences between words.  This version has 
# no pron-probs involved, and
# does support an optional silence, but this silence is only allowed
# between words (where it may occur an arbitrary number of times),
# not at the beginning or end of the file.

if(@ARGV != 2) {
    die "Usage: make_lexicon_fst_special.pl lexicon.txt silphone >lexiconfst.txt"
}

$lexfn = shift @ARGV;
$silphone = shift @ARGV;

open(L, "<$lexfn") || die "Error opening lexicon $lexfn";


$startstate = 0;
$silstate = 1;
$endstate = 2;
$nextstate = 3;

sub create_wseq {
  my $init_state = shift @_;
  my $end_state = shift @_;
  my $word_or_eps = shift @_;
  my @phones = @_;
  if (@phones == 0) { push @phones, "<eps>"; }
  my $x;
  my $curstate = $init_state;
  for ($x = 0; $x + 1 < @phones; $x++) {
    print "$curstate\t$nextstate\t$phones[$x]\t$word_or_eps\n";
    $word_or_eps = "<eps>";
    $curstate = $nextstate;
    $nextstate++;
  }
  print "$curstate\t$end_state\t$phones[$x]\t$word_or_eps\n";
}


while(<L>) {
  @A = split(" ", $_);
  $w = shift @A;
  create_wseq($startstate, $endstate, $w, @A);
  create_wseq($endstate, $endstate, $w, @A);
  create_wseq($silstate, $endstate, $w, @A);
}
print "$endstate\t$silstate\t$silphone\t<eps>\n";
print "$endstate\t0\n"; # final-cost.
