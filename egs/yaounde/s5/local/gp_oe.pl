#!/usr/bin/perl -w
# gp_oe.pl - replace œ with oe
use strict;
use warnings;
use Carp;

#BEGIN {
#    @ARGV  == 1  or croak "USAGE: reattach_apostrophes.pl FILE";
#}

use utf8;

while ( my $line = <> ) {
    chomp $line;
    $line =~ s/œ/oe/g;
    print "$line\n";
}
