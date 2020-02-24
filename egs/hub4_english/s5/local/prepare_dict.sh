#!/usr/bin/env bash

# Copyright 2010-2012 Microsoft Corporation
#           2012-2014 Johns Hopkins University (Author: Daniel Povey)
#                2015 Guoguo Chen
#                2016 Vimal Manohar

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

[ -f ./path.sh ] && . ./path.sh
. ./cmd.sh

set -e
set -o pipefail
set -u

# run this from ../
dict_suffix=
stage=-1

echo "$0 $@"  # Print the command line for logging
. utils/parse_options.sh || exit 1;

if [ $# -ne 1 ]; then
  echo "Usage: $0 <wordlist>"
  echo "e.g. : $0 data/local/local_lm/data/work/wordlist"
  exit 1
fi

wordlist=$1

dir=data/local/dict${dict_suffix}
mkdir -p $dir

if [ ! -d $dir/cmudict ]; then
  # (1) Get the CMU dictionary
  svn co  https://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict \
    $dir/cmudict || exit 1;
fi

cp $wordlist $dir/orig_wordlist

# can add -r 10966 for strict compatibility.

#(2) Dictionary preparation:


if [ $stage -le 0 ]; then
  # Make phones symbol-table (adding in silence and verbal and non-verbal noises at this point).
  # We are adding suffixes _B, _E, _S for beginning, ending, and singleton phones.

  # silence phones, one per line.
  (echo SIL; echo SPN; echo NSN; echo UNK;) > $dir/silence_phones.txt
  echo SIL > $dir/optional_silence.txt

  # nonsilence phones; on each line is a list of phones that correspond
  # really to the same base phone.
  cat $dir/cmudict/cmudict.0.7a.symbols | perl -ane 's:\r::; print;' | \
    perl -e 'while(<>){
  chop; m:^([^\d]+)(\d*)$: || die "Bad phone $_";
  $phones_of{$1} .= "$_ "; }
  foreach $list (values %phones_of) {print $list . "\n"; } ' \
    > $dir/nonsilence_phones.txt || exit 1;

  # A few extra questions that will be added to those obtained by automatically clustering
  # the "real" phones.  These ask about stress; there's also one for silence.
  cat $dir/silence_phones.txt| awk '{printf("%s ", $1);} END{printf "\n";}' > $dir/extra_questions.txt || exit 1;
  cat $dir/nonsilence_phones.txt | perl -e 'while(<>){ foreach $p (split(" ", $_)) {
  $p =~ m:^([^\d]+)(\d*)$: || die "Bad phone $_"; $q{$2} .= "$p "; } } foreach $l (values %q) {print "$l\n";}' \
    >> $dir/extra_questions.txt || exit 1;

  grep -v ';;;' $dir/cmudict/cmudict.0.7a | \
    perl -ane 'if(!m:^;;;:){ s:(\S+)\(\d+\) :$1 :; print; }' \
    > $dir/dict.cmu || exit 1;

  # Add to cmudict the silences, noises etc.

  (echo '!SIL SIL'; echo '<SPOKEN_NOISE> SPN'; echo '<unk> UNK'; echo '<NOISE> NSN'; ) | \
    cat - $dir/dict.cmu > $dir/lexicon2_raw.txt
  awk '{print $1}' $dir/lexicon2_raw.txt > $dir/wordlist_with_prons

  cat <<EOF >$dir/silence_phones.txt
SIL
SPN
NSN
UNK
EOF

fi


if [ $stage -le 2 ]; then
  if [ ! -f exp/g2p/.done ]; then
    steps/dict/train_g2p.sh --cmd "$train_cmd" \
      --silence-phones $dir/silence_phones.txt \
      $dir/dict.cmu exp/g2p
    touch exp/g2p/.done
  fi
fi

export PATH=$PATH:`pwd`/local/dict

if [ $stage -le 3 ]; then
  utils/filter_scp.pl --exclude $dir/wordlist_with_prons < $dir/orig_wordlist | \
    sort -u > $dir/oovlist
fi

if [ $stage -le 7 ]; then
  steps/dict/apply_g2p.sh --cmd "$train_cmd" \
    $dir/oovlist exp/g2p exp/g2p/oov_lex
  cat exp/g2p/oov_lex/lexicon.lex | cut -f 1,3 | awk '{if (NF > 1) print $0}' > \
    $dir/dict.oovs_g2p
fi

if [ $stage -le 8 ]; then
  # the sort | uniq is to remove a duplicated pron from cmudict.
  cat $dir/lexicon2_raw.txt $dir/dict.oovs_g2p | sort | uniq > \
    $dir/lexicon.txt || exit 1;
  # lexicon.txt is without the _B, _E, _S, _I markers.

  rm $dir/lexiconp.txt 2>/dev/null || true
fi

echo "Dictionary preparation succeeded"
