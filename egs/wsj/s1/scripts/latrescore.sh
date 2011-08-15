#!/bin/bash
# Copyright 2010-2011 Microsoft Corporation

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



if [ $# != 5 ]; then
   echo "Usage: scripts/latrescore.sh <input-decode-dir> <old-LM-fst> <new-LM-fst> <transcript-text-file> <output-decode-dir>"
   exit 1;
fi

. path.sh || exit 1;


inputdir=$1
oldlm=$2
newlm=$3
transcript=$4 # e.g. data/eval_nov92.txt
dir=$5

oldlmcommand="fstproject --project_output=true $oldlm |"
newlmcommand="fstproject --project_output=true $newlm |"

mkdir -p $dir


# First remove the old LM scores.

#lattice-lmrescore --lm-scale=-1.0 "ark:gunzip -c $inputdir/*.lats.gz|" "$oldlmcommand" \
#    ark:-  2>$dir/remove_old_lm.log | \
#lattice-lmrescore --lm-scale=1.0 ark:- "$newlmcommand" "ark,t:|gzip -c>$dir/lats.newlm.gz"  \
#   2>$dir/add_new_lm.log

for inv_acwt in 14 15 16 17 18; do
  acwt=`perl -e "print (1.0/$inv_acwt);"`;
  lattice-best-path --acoustic-scale=$acwt --word-symbol-table=data/words.txt \
    "ark:gunzip -c $dir/lats.newlm.gz|" ark,t:$dir/acwt${inv_acwt}.tra \
       2>$dir/best_path.$inv_acwt.log
  
  cat $dir/acwt${inv_acwt}.tra | \
   scripts/int2sym.pl --ignore-first-field data/words.txt | \
   sed 's:<s>::' | sed 's:</s>::' | sed 's:<UNK>::g' | \
    compute-wer --text --mode=present ark:$transcript  ark,p:-   >& $dir/wer_${inv_acwt}
done

