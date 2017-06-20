#! /usr/bin/perl

# Copyright 2017  Vimal Manohar
# Apache 2.0.

# If 'text' contains:
#  utterance1 A B C D
#  utterance2 C B
#  and you ran:
#  split_text_into_docs.pl --max-words 2 text doc2text docs
#  then 'doc2text' would contain:
#  utterance1-1 utterance1
#  utterance1-2 utterance1
#  utterance2-1 utterance2
#  and 'docs' would contain:
#  utterance1-1 A B
#  utterance1-2 C D
#  utterance2-1 C B

use warnings;
use strict;

my $max_words = 1000;

my $usage = "Usage: steps/cleanup/internal/split_text_into_docs.pl [--max-words <int>] text doc2text docs\n";

while (@ARGV > 3) {
    if ($ARGV[0] eq "--max-words") {
        shift @ARGV;
        $max_words = shift @ARGV;
    } else {
        print STDERR "$usage";
        exit (1);
    }
}

if (scalar @ARGV != 3) {
  print STDERR "$usage";
  exit (1);
}

sub min ($$) { $_[$_[0] > $_[1]] }

open TEXT, $ARGV[0] or die "$0: Could not open file $ARGV[0] for reading\n";
open DOC2TEXT, ">", $ARGV[1] or die "$0: Could not open file $ARGV[1] for writing\n";
open DOCS, ">", $ARGV[2] or die "$0: Could not open file $ARGV[2] for writing\n";

while (<TEXT>) {
  chomp;
  my @F = split;
  my $utt = shift @F;
  my $num_words = scalar @F;

  if ($num_words  <= $max_words) {
    print DOCS "$_\n";
    print DOC2TEXT "$utt $utt\n";
    next;
  }

  my $num_docs = int($num_words / $max_words) + 1;
  my $num_words_shift = int($num_words / $num_docs) + 1;
  my $words_per_doc = $num_words_shift;

  #print STDERR ("$utt num-words=$num_words num-docs=$num_docs words-per-doc=$words_per_doc\n");
  
  for (my $i = 0; $i < $num_docs; $i++) {
    my $st = $i*$num_words_shift;
    my $end = min($st + $words_per_doc, $num_words) - 1;
    print DOCS ("$utt-$i " . join(" ", @F[$st..$end]) . "\n");
    print DOC2TEXT "$utt-$i $utt\n";
  }
}
