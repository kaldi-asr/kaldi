#!/usr/bin/env perl
#qcri_buckwalter2utf8.pl - convert the qcri dictionary toutf8

use strict;
use warnings;
use Carp;

use Encode::Arabic::Buckwalter;         # imports just like 'use Encode' would, plus more

my $bw_dict = "qcri.txt";

open my $B, '<', $bw_dict or croak "Problem with $bw_dict $!";

 LINE: while ( my $line = <$B> ) {
     chomp $line;
     next LINE if ( $line =~ /^\#/ );
     my ($w,$p) = split / /, $line, 2;
     print encode 'utf8', decode 'buckwalter', $w;
     print " $p\n";
}

