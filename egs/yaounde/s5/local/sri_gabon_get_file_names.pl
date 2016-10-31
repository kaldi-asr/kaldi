#!/usr/bin/perl -w
# sri_gabon_get_file_numbers.pl - get file numbers from file names
use strict;
use warnings;
use Carp;

BEGIN {
    @ARGV == 1 or croak "USAGE: sri_gabon_get_file_numbers.pl FILENAMESLISTFILE
The input argument contains a list of the files
";
}
use File::Basename;
my @suffixlist = ( '.wav' );

while ( my $line = <> ) {
    chomp $line;
    my ($name,$path,$suffix) = fileparse($line,@suffixlist);
    print "$name\n";
}
