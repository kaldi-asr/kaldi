#! /usr/bin/perl

# Converts a kaldi integer vector in text format to 
# a kaldi vector in text format by adding a pair
# of square brackets around the data.
# Assumes the first column to be the utterance id.

while (<>) {
  chomp;
  my @F = split;

  printf ("$F[0] [ ");
  for (my $i = 1; $i <= $#F; $i++) {
    printf ("$F[$i] ");
  }
  print ("]"); 
}
