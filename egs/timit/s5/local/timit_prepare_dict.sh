#!/bin/bash

# Copyright 2013   (Authors: Daniel Povey, Bagher BabaAli)

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

# Call this script from one level above, e.g. from the s3/ directory.  It puts
# its output in data/local/.

# The parts of the output of this that will be needed are
# [in data/local/dict/ ]
# lexicon.txt
# extra_questions.txt
# nonsilence_phones.txt
# optional_silence.txt
# silence_phones.txt

# run this from ../
srcdir=data/local/data
dir=data/local/dict
lmdir=data/local/nist_lm
tmpdir=data/local/lm_tmp

mkdir -p $dir $lmdir $tmpdir

[ -f path.sh ] && . ./path.sh

#(1) Dictionary preparation:

# Make phones symbol-table (adding in silence and verbal and non-verbal noises at this point).
# We are adding suffixes _B, _E, _S for beginning, ending, and singleton phones.

# silence phones, one per line.
echo sil > $dir/silence_phones.txt
echo sil > $dir/optional_silence.txt

# nonsilence phones; on each line is a list of phones that correspond
# really to the same base phone.

# Create the lexicon, which is just an identity mapping
cut -d' ' -f2- $srcdir/train.text | tr ' ' '\n' | sort -u > $dir/phones.txt
paste $dir/phones.txt $dir/phones.txt > $dir/lexicon.txt || exit 1;
grep -v -F -f $dir/silence_phones.txt $dir/phones.txt > $dir/nonsilence_phones.txt 

# A few extra questions that will be added to those obtained by automatically clustering
# the "real" phones.  These ask about stress; there's also one for silence.
cat $dir/silence_phones.txt| awk '{printf("%s ", $1);} END{printf "\n";}' > $dir/extra_questions.txt || exit 1;
cat $dir/nonsilence_phones.txt | perl -e 'while(<>){ foreach $p (split(" ", $_)) {
  $p =~ m:^([^\d]+)(\d*)$: || die "Bad phone $_"; $q{$2} .= "$p "; } } foreach $l (values %q) {print "$l\n";}' \
 >> $dir/extra_questions.txt || exit 1;

# (2) Create the phone bigram LM
  [ -z "$IRSTLM" ] && \
    echo "LM building won't work without setting the IRSTLM env variable" && exit 1;
  ! which build-lm.sh 2>/dev/null  && \
    echo "IRSTLM does not seem to be installed (build-lm.sh not on your path): " && \
    echo "go to <kaldi-root>/tools and try 'make irstlm_tgt'" && exit 1;

  cut -d' ' -f2- $srcdir/train.text | sed -e 's:^:<s> :' -e 's:$: </s>:' \
    > $srcdir/lm_train.text
  build-lm.sh -i $srcdir/lm_train.text -n 2 -o $tmpdir/lm_phone_bg.ilm.gz

  compile-lm $tmpdir/lm_phone_bg.ilm.gz -t=yes /dev/stdout | \
  grep -v unk | gzip -c > $lmdir/lm_phone_bg.arpa.gz 

echo "Dictionary & language model preparation succeeded"
