#!/bin/bash

# Copyright      2016 Johns Hopkins University (Author: Daniel Povey);

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


# Begin configuration section.

# end configuration sections

echo "$0 $@"  # Print the command line for logging
[ -f path.sh ] && . ./path.sh


. utils/parse_options.sh

if [ $# -ne 2 ]; then
  echo "Usage: $0 [options] <input-unk-lm-fst> <lang-dir>"
  echo "e.g.: $0 exp/make_unk/unk_fst.txt data/lang_unk"
  echo ""
  echo "This script, which is called from the end of prepare_lang.sh,"
  echo "inserts the unknown-word LM FST into the lexicon FSTs"
  echo "<lang-dir>/L.fst and <lang-dir>/L_disambig.fst in place of"
  echo "the special disambiguation symbol #2 (which was inserted by"
  echo "add_lex_disambig.pl as a placeholder for this FST)."
  echo ""
  echo "  <input-unk-lm-fst>:  A text-form FST, typically with the name"
  echo "                unk_fst.txt.  We will remove all symbols from the"
  echo "                output before applying it."
  echo "  <lang-dir>:  A partially built lang/ directory.  We modify"
  echo "               L.fst and L_disambig.fst, and read only words.txt."
  exit 1;
fi


unk_lm_fst=$1
lang=$2

set -e

for f in "$unk_lm_fst" $lang/L.fst $lang/L_disambig.fst $lang/words.txt $lang/oov.int; do
  [ ! -f $f ] && echo "$0: expected file $f to exist" && exit 1;
done

unused_phone_label=$(tail -n 1 $lang/phones.txt | awk '{print $2 + 1}')
label_to_replace=$(awk '{if ($1 == "#2") {print $2;}}' <$lang/phones.txt)
! [ "$unused_phone_label" -eq "$unused_phone_label" -a "$label_to_replace" -eq "$label_to_replace" ] && \
   echo "$0: error getting unused phone label or label for #2" && exit 1


# OK, now fstreplace works based on olabels, but we actually want to deal with ilabels,
# so we need to invert all the FSTs before and after doing fstreplace.
awk '{if(NF>=4) $4 = "<eps>"; print }' <$unk_lm_fst | \
  fstcompile --isymbols=$lang/phones.txt --osymbols=$lang/words.txt | \
  fstinvert > $lang/unk_temp.fst

num_states_unk=$(fstinfo $lang/unk_temp.fst | grep '# of states' | awk '{print $NF}')

# fstreplace usage is:
# Usage: fstreplace root.fst rootlabel [rule1.fst label1 ...] [out.fst]
# ... the rootlabel should just be an otherwise unused symbol.
# all the labels are olabels (word labels).. that is hardcoded in fstreplace.

for f in L.fst L_disambig.fst; do

  # with OpenFst tools, to refer to the standard input/output you need to use
  # the empty string '' and not '-'.
  fstinvert $lang/$f | fstreplace '' "$unused_phone_label" $lang/unk_temp.fst "$label_to_replace" | fstinvert > $lang/${f}.temp

  num_states_old=$(fstinfo $lang/$f | grep '# of states' | awk '{print $NF}')
  num_states_new=$(fstinfo $lang/${f}.temp | grep '# of states' | awk '{print $NF}')
  num_states_added=$[$num_states_new-$num_states_old]
  echo "$0: in $f, substituting in the unknown-word LM (which had $num_states_unk states) added $num_states_added new FST states."
  mv -f $lang/${f}.temp $lang/$f
done

rm $lang/unk_temp.fst

exit 0;
