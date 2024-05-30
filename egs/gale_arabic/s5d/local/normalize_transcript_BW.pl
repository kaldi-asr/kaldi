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
  s/[^اأإآبتثجحخدذرزسشصضطظعغفقكلمنهويىئءؤة0-9]+/ /g;  ## Removes non Arabic or numbers
#  s/[^0-9]/ /g;
#  $_ =~ s/[^اأإآبتثجحخدذرزسشصضطظعغفقكلمنهويىئءؤة0-9]+/ /g;  ## Removes non Arabic or numbers
#  s/[0-9]+//g;
  my $BW = convertUTF8ToBuckwalter ($_);
  print OUTPUTFILE "$BW"."\n";
}
close INFILE;
close OUTPUTFILE;



# this function is copied from MADATools.pm: MADA Tools
 sub convertUTF8ToBuckwalter {

    my ($line)= (@_);
#$line = $UTF8_ENCODING_OBJ->decode($line);  ## Same as Encode::decode("utf8",$line), but faster since object already created
#$line =~ s/[^اأإآبتثجحخدذرزسشصضطظعغفقكلمنهويىئءؤة0-9]+//g;  ## Removes non Arabic or numbers
#    $line =~ s/[0-9]//g;
    $line =~ s/\x{0621}/\'/g;   ## HAMZA
    $line =~ s/\x{0622}/\|/g;   ## ALEF WITH MADDA ABOVE
    $line =~ s/\x{0623}/\>/g;   ## ALEF WITH HAMZA ABOVE
    $line =~ s/\x{0624}/\&/g;   ## WAW WITH HAMZA ABOVE
    $line =~ s/\x{0625}/\</g;   ## ALEF WITH HAMZA BELOW
    $line =~ s/\x{0626}/\}/g;   ## YEH WITH HAMZA ABOVE
    $line =~ s/\x{0627}/A/g;    ## ALEF
    $line =~ s/\x{0628}/b/g;    ## BEH
    $line =~ s/\x{0629}/p/g;    ## TEH MARBUTA
    $line =~ s/\x{062A}/t/g;    ## TEH
    $line =~ s/\x{062B}/v/g;    ## THEH
    $line =~ s/\x{062C}/j/g;    ## JEEM
    $line =~ s/\x{062D}/H/g;    ## HAH
    $line =~ s/\x{062E}/x/g;    ## KHAH
    $line =~ s/\x{062F}/d/g;    ## DAL
    $line =~ s/\x{0630}/\*/g;   ## THAL
    $line =~ s/\x{0631}/r/g;    ## REH
    $line =~ s/\x{0632}/z/g;    ## ZAIN
    $line =~ s/\x{0633}/s/g;    ## SEEN
    $line =~ s/\x{0634}/\$/g;   ## SHEEN
    $line =~ s/\x{0635}/S/g;    ## SAD
    $line =~ s/\x{0636}/D/g;    ## DAD
    $line =~ s/\x{0637}/T/g;    ## TAH
    $line =~ s/\x{0638}/Z/g;    ## ZAH
    $line =~ s/\x{0639}/E/g;    ## AIN
    $line =~ s/\x{063A}/g/g;    ## GHAIN
    $line =~ s/\x{0640}/_/g;    ## TATWEEL
    $line =~ s/\x{0641}/f/g;    ## FEH
    $line =~ s/\x{0642}/q/g;    ## QAF
    $line =~ s/\x{0643}/k/g;    ## KAF
    $line =~ s/\x{0644}/l/g;    ## LAM
    $line =~ s/\x{0645}/m/g;    ## MEEM
    $line =~ s/\x{0646}/n/g;    ## NOON
    $line =~ s/\x{0647}/h/g;    ## HEH
    $line =~ s/\x{0648}/w/g;    ## WAW
    $line =~ s/\x{0649}/Y/g;    ## ALEF MAKSURA
    $line =~ s/\x{064A}/y/g;    ## YEH

    ## Diacritics
    $line =~ s/\x{064B}/F/g;    ## FATHATAN
    $line =~ s/\x{064C}/N/g;    ## DAMMATAN
    $line =~ s/\x{064D}/K/g;    ## KASRATAN
    $line =~ s/\x{064E}/a/g;    ## FATHA
    $line =~ s/\x{064F}/u/g;    ## DAMMA
    $line =~ s/\x{0650}/i/g;    ## KASRA
    $line =~ s/\x{0651}/\~/g;   ## SHADDA
    $line =~ s/\x{0652}/o/g;    ## SUKUN
    $line =~ s/\x{0670}/\`/g;   ## SUPERSCRIPT ALEF

    $line =~ s/\x{0671}/\{/g;   ## ALEF WASLA
    $line =~ s/\x{067E}/P/g;    ## PEH
    $line =~ s/\x{0686}/J/g;    ## TCHEH
    $line =~ s/\x{06A4}/V/g;    ## VEH
    $line =~ s/\x{06AF}/G/g;    ## GAF


    ## Punctuation should really be handled by the utf8 cleaner or other method
#   $line =~ s/\xa2/\,/g; # comma
#    $line =~ s//\,/g; # comma
#    $line =~ s//\,/g;
#    $line =~ s//\;/g; # semicolon
#    $line =~ s//\?/g; # questionmark

    return $line;
}
