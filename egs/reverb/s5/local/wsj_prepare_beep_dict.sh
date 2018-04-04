#!/bin/bash

# Copyright 2013 MERL (author: Felix Weninger)
# Contains some code by Microsoft Corporation, Johns Hopkins University (author: Daniel Povey)

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
# WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
# MERCHANTABLITY OR NON-INFRINGEMENT.
# See the Apache 2 License for the specific language governing permissions and
# limitations under the License.

# run this from ../
dir=data/local/dict
mkdir -p $dir

# Get BEEP dictionary
BEEP_URL="https://www.dropbox.com/s/skiemso2ohw1ood/beep-1.0.tar.gz"
x=`basename $BEEP_URL`
mkdir -p $dir/beep
if [ ! -e $dir/beep/$x ]; then
  wget $BEEP_URL -O $dir/beep/$x || exit 1;
  tar zxvf $dir/beep/$x -C $dir || exit 1;
fi

# (1) Get the CMU dictionary
if [ ! -d $dir/cmudict/.svn ]; then
  svn co  https://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict \
    $dir/cmudict || exit 1;
fi

# can add -r 10966 for strict compatibility.

# merge pronunciations 
# so that beep pronunciations take precedence
perl local/merge_dict.pl $dir/beep/beep-1.0 $dir/cmudict/cmudict.0.7a $dir/beep_cmu.dict

#(2) Dictionary preparation:


# Make phones symbol-table (adding in silence and verbal and non-verbal noises at this point).
# We are adding suffixes _B, _E, _S for beginning, ending, and singleton phones.

# silence phones, one per line.
(echo SIL; echo SPN; echo NSN) > $dir/silence_phones.txt
echo SIL > $dir/optional_silence.txt

# obtain list of phones
grep -vi sil $dir/beep_cmu.dict | \
perl -e 'while(<>){
  next if /^#/;
  my @e=split;
  for (@e[1..$#e]) { $p{$_}=1; }
}
print join("\n", map { uc } sort keys %p), "\n"'  \
> $dir/nonsilence_phones.txt || exit 1;
# nonsilence phones; on each line is a list of phones that correspond
# really to the same base phone.
#cat $dir/cmudict/cmudict.0.7a.symbols | perl -ane 's:\r::; print;' | \
# perl -e 'while(<>){
#  chop; m:^([^\d]+)(\d*)$: || die "Bad phone $_"; 
#  $phones_of{$1} .= "$_ "; }
#  foreach $list (values %phones_of) {print $list . "\n"; } ' \
#  > $dir/nonsilence_phones.txt || exit 1;

# A few extra questions that will be added to those obtained by automatically clustering
# the "real" phones.  These ask about stress; there's also one for silence.
cat $dir/silence_phones.txt| awk '{printf("%s ", $1);} END{printf "\n";}' > $dir/extra_questions.txt || exit 1;
cat $dir/nonsilence_phones.txt | perl -e 'while(<>){ foreach $p (split(" ", $_)) {
  $p =~ m:^([^\d]+)(\d*)$: || die "Bad phone $_"; $q{$2} .= "$p "; } } foreach $l (values %q) {print "$l\n";}' \
 >> $dir/extra_questions.txt || exit 1;

#exit

grep -v "^#" $dir/beep_cmu.dict > $dir/lexicon1_raw_nosil.txt
#grep -v ';;;' $dir/cmudict/cmudict.0.7a | \
# perl -ane 'if(!m:^;;;:){ s:(\S+)\(\d+\) :$1 :; print; }' \
#  > $dir/lexicon1_raw_nosil.txt || exit 1;


# Add to cmudict the silences, noises etc.
(echo '!SIL SIL'; echo '<SPOKEN_NOISE> SPN'; echo '<UNK> SPN'; echo '<NOISE> NSN'; ) | \
 cat - $dir/lexicon1_raw_nosil.txt  > $dir/lexicon2_raw.txt || exit 1;


# lexicon.txt is without the _B, _E, _S, _I markers.
cp $dir/lexicon2_raw.txt $dir/lexicon.txt


echo "Dictionary preparation succeeded"
