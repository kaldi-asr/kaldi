#!/usr/bin/perl -w
#remove-punctuation.pl - remove punctuation and other characters

use strict;
use warnings;
use Carp;
use Encode;

binmode STDOUT, ":utf8";

while ( my $line = <> ) {

    $line = decode_utf8 $line;

    chomp $line;

#remove control characters
    $line =~ s/(\p{Other})/ /g;
    $line =~ s/(\p{Control})/ /g;
    $line =~ s/(\p{Format})/ /g;
    $line =~ s/(\p{Private_Use})/ /g;
    $line =~ s/(\p{Surrogate})/ /g;

#marks
    #$line =~ s/(\p{Mark})/ /g;
    #$line =~ s/(\p{Non_Spacing_Mark})/ /g;
#$line =~ s/(\p{Spacing_Combining_Mark})/ /g;
    #$line =~ s/(\p{Enclosing_Mark})/ /g;

#punctuation
    #$line =~ s/(\p{Punctuation})/ /g;
#$line =~ s/(\p{Dash_Punctuation})/ /g;
    #$line =~ s/(\p{Close_Punctuation})/ /g;
    #$line =~ s/(\p{Open_Punctuation})/ /g;
    #$line =~ s/(\p{Initial_Punctuation})/ /g;
    $line =~ s/(\p{Final_Punctuation})/ /g;
    #$line =~ s/(\p{Connector_Punctuation})/ /g;
    #$line =~ s/(\p{Other_Punctuation})/ /g;
    $line =~ s/[.;:?,]//g;

#symbols
    #$line =~ s/(\p{Symbol})/ /g;
    $line =~ s/(\p{Math_Symbol})/ /g;
    $line =~ s/(\p{Currency_Symbol})/ /g;
    $line =~ s/(\p{Modifier_Symbol})/ /g;
    $line =~ s/(\p{Other_Symbol})/ /g;

#quotes
    # $line =~ s/(\p{QMark})/ /g;
    $line =~ s/ \" / /g;
    $line =~ s/ \"$/ /g;
    $line =~ s/^\" //g;

    # remove dashes
    $line =~ s/ \- / /g;

    # remove parens
    $line =~ s/\(//g;
    $line =~ s/\)//g;
    
    # remove exclamation marks
    $line =~ s/ \! / /g;
    $line =~ s/ \!$/ /g;

    print "$line\n";

}

