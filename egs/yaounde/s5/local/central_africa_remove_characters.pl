#!/usr/bin/perl -w
#central_remove-punctuation.pl - remove punctuation and other characters

use strict;
use warnings;
use Carp;
use Encode;

binmode STDOUT, ":utf8";

while ( my $line = <> ) {

    $line = decode_utf8 $line;

    chomp $line;
    my ($n,$sent) = split /\t/, $line, 2;

    print "$n\t";

#remove control characters
    #$sent =~ s/(\p{Other})/ /g;
    #$sent =~ s/(\p{Control})/ /g;
    #$sent =~ s/(\p{Format})/ /g;
    #$sent =~ s/(\p{Private_Use})/ /g;
    #$sent =~ s/(\p{Surrogate})/ /g;

#marks
    #$sent =~ s/(\p{Mark})/ /g;
    #$sent =~ s/(\p{Non_Spacing_Mark})/ /g;
#$sent =~ s/(\p{Spacing_Combining_Mark})/ /g;
    #$sent =~ s/(\p{Enclosing_Mark})/ /g;

#punctuation
    #$sent =~ s/(\p{Punctuation})/ /g;
#$sent =~ s/(\p{Dash_Punctuation})/ /g;
    #$sent =~ s/(\p{Close_Punctuation})/ /g;
    #$sent =~ s/(\p{Open_Punctuation})/ /g;
    #$sent =~ s/(\p{Initial_Punctuation})/ /g;
    #$sent =~ s/(\p{Final_Punctuation})/ /g;
    $sent =~ s/[.;:?]//;
    #$sent =~ s/(\p{Connector_Punctuation})/ /g;
    #$sent =~ s/(\p{Other_Punctuation})/ /g;

#symbols
    #$sent =~ s/(\p{Symbol})/ /g;
    #$sent =~ s/(\p{Math_Symbol})/ /g;
    #$sent =~ s/(\p{Currency_Symbol})/ /g;
    #$sent =~ s/(\p{Modifier_Symbol})/ /g;
    #$sent =~ s/(\p{Other_Symbol})/ /g;

#quotes
    # $sent =~ s/(\p{QMark})/ /g;

    print "$sent\n";

}

