#!/usr/bin/perl -w
# reattach_apostrophes.pl - fix orphan apostrophes
use strict;
use warnings;
use Carp;

#BEGIN {
#    @ARGV == 1 or croak "USAGE: reattach_apostrophes.pl FILE";
#}

while ( my $line = <> ) {
    chomp $line;
    $line =~ s/([cdjlnst]) ' /$1' /g;
    $line =~ s/(lorsqu) ' /$1' /g;

    $line =~ s/ qu ' / qu' /g;

        print "$line\n";
}
