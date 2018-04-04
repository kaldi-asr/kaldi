#!/usr/bin/env perl

use strict;

my $utt;
my %text;
while (<>) {
    next if (/^#!/);
    s/\x0D$//;
    chomp;
    if (/"\*\/\*?(\w+)\*?\.lab"/) {
    #if (/"(.+)"/) {
        $utt = lc $1;
        #print "utterance: $utt\n";
    } 
    elsif (!/^\.$/) {
        if ($text{$utt}) {
            $text{$utt} .= " ";
        }
        $text{$utt} .= $_;
    }
}

for my $utt (sort keys %text) 
{
    print "$utt $text{$utt}\n";
}
