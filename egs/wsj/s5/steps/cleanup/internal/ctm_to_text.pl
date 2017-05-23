#! /usr/bin/perl

# Copyright 2016  Vimal Manohar
# Apache 2.0.

use strict;
use warnings;

if (scalar @ARGV != 1 && scalar @ARGV != 3) {
  my $usage = <<END;
This script converts a CTM into kaldi text format by concatenating the words
belonging to the same utterance (or recording) and outputs the same to the
standard output.
If --non-scored-words list file is provided with one word per line, then 
those words are not added to the text.

The CTM format is <file> <channel> <start-time> <duration> <word> [<conf>].
This script assumes the CTM to be in NIST sorted order given by UNIX
sort command "sort +0 -1 +1 -2 +2nb -3"

Usage: ctm_to_text.pl [--non-scored-words <file>] <ctm-file> > <text>
END
  die $usage;
}

my $non_scored_words_list = "";
if (scalar @ARGV > 1) {
  if ($ARGV[0] eq "--non-scored-words") {
    shift @ARGV;
    $non_scored_words_list = shift @ARGV;
  } else {
    die "Unknown option $ARGV[0]\n";
  }
}

my %non_scored_words;
$non_scored_words{"<eps>"} = 1;

if ($non_scored_words_list ne "") {
  open NONSCORED, $non_scored_words_list or die "Failed to open $non_scored_words_list";
  
  while (<NONSCORED>) {
    chomp;
    my @F = split;
    $non_scored_words{$F[0]} = 1;
  }

  close NONSCORED;
}

my $ctm_file = shift @ARGV;
open CTM, $ctm_file or die "Failed to open $ctm_file";

my $prev_utt = "";
my @text;

while (<CTM>) {
  chomp;
  my @F = split;

  my $utt = $F[0];
  if ($utt ne $prev_utt && $prev_utt ne "") {
    if (scalar @text > 0) {
      print $prev_utt . " " . join(" ", @text) . "\n";
    }
    @text = ();
  }
  
  if (scalar @F < 5 || scalar @F > 6) {
    die "Invalid line $_ in CTM $ctm_file\n";
  }

  if (!defined $non_scored_words{$F[4]}) {
    push @text, $F[4];
  }

  $prev_utt = $utt;
}

close CTM;
    
if (scalar @text > 0) {
  print $prev_utt . " " . join(" ", @text) . "\n";
}
