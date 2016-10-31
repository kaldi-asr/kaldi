#!/usr/bin/perl -w
#get-utt2text.pl - make the text file
use strict;
use warnings;
use Carp;
BEGIN {
    @ARGV == 1 or croak "get-utt2text.pl PROMPTSFILE";
}

use File::Basename;

my ($promptsfile) = @ARGV;

while ( my $line = <>) {
    chomp $line;
    my ($pth, $prmpt) = split /\s+/, $line, 2;
    my $utt = basename $pth, ".wav";
    print "$utt\t$prmpt\n";
}
