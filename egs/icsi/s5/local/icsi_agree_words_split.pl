#!/usr/bin/perl

# Pawel Swietojanski, 2014
# Apache 2.0

if (@ARGV != 3) {
  print STDERR "Usage: icsi_agree_words_split.pl <split-file> <in-segments> <out-segments>\n";
  exit(1);
}

my $split_file=shift @ARGV;
my $seg_in=shift @ARGV;
my $seg_out=shift @ARGV;

open(RF, "<$split_file") || die "opening replace file $split_file";
open(SI, "<$seg_in") || die "opening input segment file $seg_in";
open(SO, ">$seg_out") || die "opening output segment file $seg_out";

my @segments = <SI>;
my $segments = join("", @segments);

while(<RF>) {
  my $to_rep = $_; chomp($to_rep);
  my $words = join(" ", split("-", $to_rep));
  print "Replacing $to_rep with $words\n";
  $segments =~ s/\s+$to_rep(\s+)/ $words\1/g;
}

print SO $segments;

close(RF);
close(SI);
close(SO);

