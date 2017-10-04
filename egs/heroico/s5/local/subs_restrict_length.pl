#!/usr/bin/perl -w
# subs_restrict_length.pl - restrict length of segments

use strict;
use warnings;
use Carp;

BEGIN {
    @ARGV == 1 or croak "USAGE: $0 <PATH_TO_SUBS_CORPUS>
	$0 /mnt/corpora/subs/OpenSubtitles2016.ar-en.ar
	";
}

use Encode;

# set lower and upper bounds
my $lb = 8;
my $ub = 16;

# input and output files
my ($c) = @ARGV;

my $o = "data/local/tmp/subs/lm/es.txt";

open my $C, '<', $c or croak "problems with $c $!";

system "mkdir -p data/local/tmp/subs/lm";

open my $O, '+>:utf8', $o or croak "problems with $o $!";

LINE: while ( my $line = <$C> ) {

    $line = decode_utf8 $line;

    chomp $line;

    my @tokens = split /\s+/, $line;

    next LINE if ( ($#tokens < $lb) or ($#tokens > $ub ));

    print $O "$line\n";

}

close $C;
close $O;
