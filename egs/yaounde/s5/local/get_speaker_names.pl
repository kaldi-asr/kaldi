#!/usr/bin/perl -w
# get-speakernames.pl - write the directory name for each speaker
use strict;
use warnings;
use Carp;

BEGIN {
    @ARGV == 2 or croak "USAGE: get-speakernames.pl SPEAKERDIRPATHSFILE FOLD";
}

use File::Basename;

my ($dirnames,$fold) = @ARGV;

open my $FH, '<', "$dirnames" or croak "could not open file $dirnames for reading $!";

while ( my $line = <$FH>) {
    chomp $line;
    my $b = basename $line;
    print "$b\n";
}


