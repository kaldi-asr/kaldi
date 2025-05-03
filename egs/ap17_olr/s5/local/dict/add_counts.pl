#!/usr/bin/env perl


# Add counts to an oovlist.
# Reads in counts as output by uniq -c, and
# an oovlist, and prints out the counts of the oovlist.

(@ARGV == 1 || @ARGV == 2) || die "Usage: add_counts.pl count_file [oovlist]\n";

$counts = shift @ARGV;

open(C, "<$counts") || die "Opening counts file $counts";

while(<C>) {
  @A = split(" ", $_);
  @A == 2 || die "Bad line in counts file: $_";
  ($count, $word) = @A;
  $count =~ m:^\d+$: || die "Bad count $A[0]\n";
  $counts{$word} = $count;
}

while(<>) {
  chop;
  $w = $_;
  $w =~ m:\S+: || die "Bad word $w";
  defined $counts{$w} || die "Word $w not present in counts file";
  print "\t$counts{$w}\t$w\n";
}
    
  

