#!/usr/bin/env perl

# Copyright 2014 QCRI (author: Ahmed Ali)
# Apache 2.0

use warnings;
use strict;
use Encode;
use utf8;



if (@ARGV !=2 )
    {#
	print "usage: $0 <inFile> <onlyArabicFile>\n"; 
	exit (1);   
    }
    
# <\check usage>
my $inFile = shift (@ARGV);
my $ouFile = shift(@ARGV);


open INFILE, "<$inFile" || die "unable to open the input file $inFile\n";
binmode INFILE, ":encoding(utf8)";


open OUTPUTFILE, ">$ouFile" or die "unable to open the output mlf file $ouFile\n";
binmode OUTPUTFILE, ":encoding(utf8)";

while (<INFILE>) {
  my $BW = convertUTF8ToBuckwalter ($_);
  print OUTPUTFILE "$BW";
}
close INFILE;
close OUTPUTFILE;



# this function is copied from MADATools.pm: MADA Tools
 sub convertUTF8ToBuckwalter {

    my ($line)= (@_);
    $line =~ s/\'/\x{0621}/g;   ## HAMZA
    $line =~ s/\|/\x{0622}/g;   ## ALEF WITH MADDA ABOVE
    $line =~ s/\>/\x{0623}/g;   ## ALEF WITH HAMZA ABOVE
    $line =~ s/\&/\x{0624}/g;   ## WAW WITH HAMZA ABOVE
    $line =~ s/\</\x{0625}/g;   ## ALEF WITH HAMZA BELOW
    $line =~ s/\}/\x{0626}/g;   ## YEH WITH HAMZA ABOVE
    $line =~ s/A/\x{0627}/g;    ## ALEF
    $line =~ s/b/\x{0628}/g;    ## BEH
    $line =~ s/p/\x{0629}/g;    ## TEH MARBUTA
    $line =~ s/t/\x{062A}/g;    ## TEH
    $line =~ s/v/\x{062B}/g;    ## THEH
    $line =~ s/j/\x{062C}/g;    ## JEEM
    $line =~ s/H/\x{062D}/g;    ## HAH
    $line =~ s/x/\x{062E}/g;    ## KHAH
    $line =~ s/d/\x{062F}/g;    ## DAL
    $line =~ s/\*/\x{0630}/g;   ## THAL
    $line =~ s/r/\x{0631}/g;    ## REH
    $line =~ s/z/\x{0632}/g;    ## ZAIN
    $line =~ s/s/\x{0633}/g;    ## SEEN
    $line =~ s/\$/\x{0634}/g;   ## SHEEN
    $line =~ s/S/\x{0635}/g;    ## SAD
    $line =~ s/D/\x{0636}/g;    ## DAD
    $line =~ s/T/\x{0637}/g;    ## TAH
    $line =~ s/Z/\x{0638}/g;    ## ZAH
    $line =~ s/E/\x{0639}/g;    ## AIN
    $line =~ s/g/\x{063A}/g;    ## GHAIN
    $line =~ s/_/\x{0640}/g;    ## TATWEEL
    $line =~ s/f/\x{0641}/g;    ## FEH
    $line =~ s/q/\x{0642}/g;    ## QAF
    $line =~ s/k/\x{0643}/g;    ## KAF
    $line =~ s/l/\x{0644}/g;    ## LAM
    $line =~ s/m/\x{0645}/g;    ## MEEM
    $line =~ s/n/\x{0646}/g;    ## NOON
    $line =~ s/h/\x{0647}/g;    ## HEH
    $line =~ s/w/\x{0648}/g;    ## WAW
    $line =~ s/Y/\x{0649}/g;    ## ALEF MAKSURA
    $line =~ s/y/\x{064A}/g;    ## YEH

    ## Diacritics
    $line =~ s/F/\x{064B}/g;    ## FATHATAN
    $line =~ s/N/\x{064C}/g;    ## DAMMATAN
    $line =~ s/K/\x{064D}/g;    ## KASRATAN
    $line =~ s/a/\x{064E}/g;    ## FATHA
    $line =~ s/u/\x{064F}/g;    ## DAMMA
    $line =~ s/i/\x{0650}/g;    ## KASRA
    $line =~ s/\~/\x{0651}/g;   ## SHADDA
    $line =~ s/o/\x{0652}/g;    ## SUKUN
    $line =~ s/\`/\x{0670}/g;   ## SUPERSCRIPT ALEF

    $line =~ s/\{/\x{0671}/g;   ## ALEF WASLA
    $line =~ s/P/\x{067E}/g;    ## PEH
    $line =~ s/J/\x{0686}/g;    ## TCHEH
    $line =~ s/V/\x{06A4}/g;    ## VEH
    $line =~ s/G/\x{06AF}/g;    ## GAF


    ## Punctuation should really be handled by the utf8 cleaner or other method






    return $line;
}
