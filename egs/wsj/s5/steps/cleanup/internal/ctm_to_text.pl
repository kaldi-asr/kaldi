#! /usr/bin/perl

# Copyright 2016  Vimal Manohar
# Apache 2.0.

use strict;
use warnings;

my $prev_utt = "";
my @text;

my $non_scored_words_list = "";
if (scalar @ARGV > 0) {
  if ($ARGV[0] eq "--non-scored-words") {
    shift @ARGV;
    $non_scored_words_list = shift @ARGV;
  }
}

my %non_scored_words;
$non_scored_words{"<eps>"} = 1;

if ($non_scored_words_list ne "") {
  open NONSCORED, $non_scored_words_list or die "Failed to open $non_scored_words_list";
  
  while (<NONSCORED>) {
    chomp;
    $non_scored_words{$_} = 1;
  }
}

while (<>) {
  chomp;
  my @F = split;

  my $utt = $F[0];
  if ($utt ne $prev_utt && $prev_utt ne "") {
    if (scalar @text > 0) {
      print $prev_utt . " " . join(" ", @text) . "\n";
    }
    @text = ();
  }
  if (!defined $non_scored_words{$F[4]}) {
    push @text, $F[4];
  }

  $prev_utt = $utt;
}
    
if (scalar @text > 0) {
  print $prev_utt . " " . join(" ", @text) . "\n";
}
