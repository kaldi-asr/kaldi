#!/usr/bin/perl

# Pawel Swietojanski, 2014
# Apache 2.0

if (@ARGV != 3) {
  print STDERR "Usage: icsi_agree_words.pl <replace-file> <in-segments> <out-segments>\n";
  exit(1);
}

my $replace_file=shift @ARGV;
my $seg_in=shift @ARGV;
my $seg_out=shift @ARGV;

open(RF, "<$replace_file") || die "opening replace file $replace_file";
open(SI, "<$seg_in") || die "opening input segment file $seg_in";
open(SO, ">$seg_out") || die "opening output segment file $seg_out";

my @segments =  <SI>;
my $segments = join("", @segments);

while(<RF>) {
  my @A = split(/\s+/, $_);
  $#A>0 || die "Incorrect number of columns $#A in $replace_file";
  my $word1 = @A[0];
  my $word2 = join(" ", @A[1..$#A]);
  chomp($word1); chomp($word2);
  print "Replacing $word1 with $word2\n";  
  $segments =~ s/\s+$word1(\s+)/ $word2\1/g;
}

print SO $segments;

close(RF);
close(SI);
close(SO);

