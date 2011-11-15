#!/bin/bash

# This script takes the dictionary prepared in wsj_extend_dict.sh
# (which puts its output in data/local/dict_larger), and puts
# in data/local/dict_larger_prep a file "lexicon.txt" that
# contains the begin/end markings and silence, etc.  There
# is also a file phones_disambig.txt in the same directory.
# 
# This script is similar to wsj_prepare_dict.sh

# Call this script from one level above, e.g. from the s3/ directory.  It puts
# its output in data/local/.

dictin=data/local/dict_larger/dict.final
[ ! -f $dictin ] && echo No such file $dictin && exit 1;

# run this from ../
dir=data/local/dict_larger_prep
mkdir -p $dir

# Make phones symbol-table (adding in silence and verbal and non-verbal noises at this point).
# We are adding suffixes _B, _E, _S for beginning, ending, and singleton phones.

cat data/local/cmudict/cmudict.0.7a.symbols | perl -ane 's:\r::; print;' | \
 awk 'BEGIN{print "<eps> 0"; print "SIL 1"; print "SPN 2"; print "NSN 3"; N=4; } 
           {printf("%s %d\n", $1, N++); }
           {printf("%s_B %d\n", $1, N++); }
           {printf("%s_E %d\n", $1, N++); }
           {printf("%s_S %d\n", $1, N++); } ' >$dir/phones.txt


# First make a version of the lexicon without the silences etc, but with the position-markers.
# Remove the comments from the cmu lexicon and remove the (1), (2) from words with multiple 
# pronunciations.

cat $dictin | \
 perl -ane '@A=split(" ",$_); $w = shift @A; @A>0||die;
   if(@A==1) { print "$w $A[0]_S\n"; } else { print "$w $A[0]_B ";
     for($n=1;$n<@A-1;$n++) { print "$A[$n] "; } print "$A[$n]_E\n"; } ' \
  > $dir/lexicon_nosil.txt || exit 1;

# Add the silences, noises etc.

(echo '!SIL SIL'; echo '<SPOKEN_NOISE> SPN'; echo '<UNK> SPN'; echo '<NOISE> NSN'; ) | \
 cat - $dir/lexicon_nosil.txt  > $dir/lexicon.txt || exit 1;

echo "Local dictionary preparation succeeded; output is in $dir/lexicon.txt"

