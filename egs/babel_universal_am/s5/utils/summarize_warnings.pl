#!/usr/bin/env perl

# Copyright 2012 Johns Hopkins University (Author: Daniel Povey).  Apache 2.0.

 @ARGV != 1 && print STDERR "Usage: summarize_warnings.pl <log-dir>\n" && exit 1;

$dir = $ARGV[0];

! -d $dir && print STDERR "summarize_warnings.pl: no such directory $dir\n" && exit 1;

$dir =~ s:/$::; # Remove trailing slash.


# Group the files into categories where all have the same base-name.
foreach $f (glob ("$dir/*.log")) {
  $f_category = $f;
  # do next expression twice; s///g doesn't work as they overlap.
  $f_category =~ s:\.\d+\.:.*.:;
  $f_category =~ s:\.\d+\.:.*.:;
  $fmap{$f_category} .= " $f";
}

sub split_hundreds { # split list of filenames into groups of 100.
  my $names = shift @_;
  my @A = split(" ", $names);
  my @ans = ();
  while (@A > 0) {
    my $group = "";
    for ($x = 0; $x < 100 && @A>0; $x++) {
      $fname = pop @A;
      $group .= "$fname ";
    }
    push @ans, $group;
  }
  return @ans;
}

foreach $c (keys %fmap) {
  $n = 0;
  foreach $fgroup (split_hundreds($fmap{$c})) {
    $n += `grep -w WARNING $fgroup | wc -l`;
  }
  if ($n != 0) {
    print "$n warnings in $c\n"
  }
}
