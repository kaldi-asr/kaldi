#!/usr/bin/env bash
# Copyright (c) 2013, Ondrej Platek, Ufal MFF UK <oplatek@ufal.mff.cuni.cz>
#
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
# limitations under the License. #
locdata=$1; shift
train_text=$1; shift
test_text=$1; shift
local_lm=$1; shift
lms=$1; shift


mkdir -p $local_lm

echo "=== Preparing the LM ..."

function build_0gram {
    echo "=== Building zerogram $lm from ${transcr}. ..."
    transcr=$1; lm=$2
    cut -d' ' -f2- $transcr | tr ' ' '\n' | sort -u > $lm
    echo "<s>" >> $lm
    echo "</s>" >> $lm
    python -c """
import math
with open('$lm', 'r+') as f:
    lines = f.readlines()
    p = math.log10(1/float(len(lines)));
    lines = ['%f\\t%s'%(p,l) for l in lines]
    f.seek(0); f.write('\\n\\\\data\\\\\\nngram  1=       %d\\n\\n\\\\1-grams:\\n' % len(lines))
    f.write(''.join(lines) + '\\\\end\\\\')
"""
}

for lm in $lms ; do
    lm_base=`basename $lm`
    if [ ${lm_base%[0-6]} !=  'build' ] ; then
        cp $lm $local_lm
    else
        # We will build the LM 'build[0-9].arpa
        lm_order=${lm_base#build}

        echo "=== Building LM of order ${lm_order}..."
        if [ $lm_order -eq 0 ] ; then
            echo "Zerogram $lm_base LM is build from text: $test_text"
            cut -d' ' -f2- $test_text | sed -e 's:^:<s> :' -e 's:$: </s>:' | \
                sort -u > $locdata/lm_test.txt
            build_0gram  $locdata/lm_test.txt $local_lm/${lm_base}
        else
            echo "LM $lm_base is build from text: $train_text"
            cut -d' ' -f2- $train_text | sed -e 's:^:<s> :' -e 's:$: </s>:' | \
                sort -u > $locdata/lm_train.txt
            ngram-count -text $locdata/lm_train.txt -order ${lm_order} \
                -wbdiscount -interpolate -lm $local_lm/${lm_base}
        fi
    fi
done
echo "*** LMs preparation finished!"

echo "=== Preparing the vocabulary ..."

if [ "$DICTIONARY" == "build" ]; then
  echo; echo "Building dictionary from train data"; echo
  cut -d' ' -f2- $train_text | tr ' ' '\n' > $locdata/vocab-full-raw.txt
else
  echo; echo "Using predefined dictionary: ${DICTIONARY}"
  echo "Throwing away first 2 rows."; echo
  tail -n +3 $DICTIONARY | cut -f 1 > $locdata/vocab-full-raw.txt
fi

echo '</s>' >> $locdata/vocab-full-raw.txt
echo "Removing from vocabulary _NOISE_, and  all '_' words from vocab-full.txt"
cat $locdata/vocab-full-raw.txt | grep -v '_' | \
  sort -u > $locdata/vocab-full.txt
echo "*** Vocabulary preparation finished!"


echo "Removing from vocabulary _NOISE_, and  all '_' words from vocab-test.txt"
cut -d' ' -f2 $test_text | tr ' ' '\n' | grep -v '_' | sort -u > $locdata/vocab-test.txt

