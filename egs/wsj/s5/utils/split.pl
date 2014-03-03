#!/usr/bin/perl
#
# Copyright 2012 Johns Hopkins University (Author: Yenda Trmal).  Apache 2.0.

@ARGV != 2 && print STDERR "Usage: split.pl -n l/n/N \n" && exit 1;

$first_flag=$ARGV[0];

die  "Usage: split.pl -n l/n/N \n" if $ARGV[0] ne "-n";

($l, $n, $MAX) = split("/", $ARGV[1]);

die  "Usage: split.pl -n l/n/N \n" if $l ne "l";

$line_num=0;
$n=$n-1;
while ( $line=<STDIN>) {
  print $line if ($line_num % $MAX) == $n;
  $line_num+=1;
}

