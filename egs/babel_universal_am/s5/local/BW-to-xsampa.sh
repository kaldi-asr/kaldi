!/bin/bash

# Pronunciation map grapheme to xsampa
#$line =~ s/\x{0621}/\'/g;   ## HAMZA ?
#$line =~ s/\x{0622}/\|/g;   ## ALEF WITH MADDA ABOVE a:
#$line =~ s/\x{0623}/\>/g;   ## ALEF WITH HAMZA ABOVE a
#$line =~ s/\x{0624}/\&/g;   ## WAW WITH HAMZA ABOVE u:
#$line =~ s/\x{0625}/\</g;   ## ALEF WITH HAMZA BELOW i
#$line =~ s/\x{0626}/\}/g;   ## YEH WITH HAMZA ABOVE i:
#$line =~ s/\x{0627}/A/g;    ## ALEF a:
#$line =~ s/\x{0628}/b/g;    ## BEH b
#$line =~ s/\x{0629}/p/g;    ## TEH MARBUTA a / a t
#$line =~ s/\x{062A}/t/g;    ## TEH t
#$line =~ s/\x{062B}/v/g;    ## THEH T
#$line =~ s/\x{062C}/j/g;    ## JEEM Z
#$line =~ s/\x{062D}/H/g;    ## HAH X\
#$line =~ s/\x{062E}/x/g;    ## KHAH X 
#$line =~ s/\x{062F}/d/g;    ## DAL d
#$line =~ s/\x{0630}/\*/g;   ## THAL D
#$line =~ s/\x{0631}/r/g;    ## REH r
#$line =~ s/\x{0632}/z/g;    ## ZAIN z
#$line =~ s/\x{0633}/s/g;    ## SEEN s
#$line =~ s/\x{0634}/\$/g;   ## SHEEN S
#$line =~ s/\x{0635}/S/g;    ## SAD s_?\
#$line =~ s/\x{0636}/D/g;    ## DAD d_?\
#$line =~ s/\x{0637}/T/g;    ## TAH t_?\
#$line =~ s/\x{0638}/Z/g;    ## ZAH D_?\
#$line =~ s/\x{0639}/E/g;    ## AIN ?\
#$line =~ s/\x{063A}/g/g;    ## GHAIN R\
#$line =~ s/\x{0640}/_/g;    ## TATWEEL 
#$line =~ s/\x{0641}/f/g;    ## FEH f
#$line =~ s/\x{0642}/q/g;    ## QAF q
#$line =~ s/\x{0643}/k/g;    ## KAF k
#$line =~ s/\x{0644}/l/g;    ## LAM l
#$line =~ s/\x{0645}/m/g;    ## MEEM m
#$line =~ s/\x{0646}/n/g;    ## NOON n
#$line =~ s/\x{0647}/h/g;    ## HEH h
#$line =~ s/\x{0648}/w/g;    ## WAW w / u:
#$line =~ s/\x{0649}/Y/g;    ## ALEF MAKSURA a
#$line =~ s/\x{064A}/y/g;    ## YEH j / i:
#
### Diacritics
#$line =~ s/\x{064B}/F/g;    ## FATHATAN an
#$line =~ s/\x{064C}/N/g;    ## DAMMATAN un
#$line =~ s/\x{064D}/K/g;    ## KASRATAN in
#$line =~ s/\x{064E}/a/g;    ## FATHA a
#$line =~ s/\x{064F}/u/g;    ## DAMMA u
#$line =~ s/\x{0650}/i/g;    ## KASRA i
#$line =~ s/\x{0651}/\~/g;   ## SHADDA
#$line =~ s/\x{0652}/o/g;    ## SUKUN
#$line =~ s/\x{0670}/\`/g;   ## SUPERSCRIPT ALEF
#
#$line =~ s/\x{0671}/\{/g;   ## ALEF WASLA
#$line =~ s/\x{067E}/P/g;    ## PEH p
#$line =~ s/\x{0686}/J/g;    ## TCHEH 
#$line =~ s/\x{06A4}/V/g;    ## VEH v
#$line =~ s/\x{06AF}/G/g;    ## GAF g


    ## Punctuation should really be handled by the utf8 cleaner or other method
#   $line =~ s/\xa2/\,/g; # comma
#    $line =~ s//\,/g; # comma
#    $line =~ s//\,/g;
#    $line =~ s//\;/g; # semicolon
#    $line =~ s//\?/g; # questionmark


