#!/bin/bash
# Copyright 2012  Brno University of Technology (author: Karel Vesely)

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

# To be run from ..


if [ $# != 4 ]; then
   echo "Usage: $0 <ali-tag> <ali-rspecifier> <transition-model> <lang>"
   echo " e.g.: $0 'TRAINING SET' 'ark:gunzip -c \$alidir/ali.gz |' tri1/final.mdl "
   exit 1;
fi

if [ -f path.sh ]; then . path.sh; fi

tag=$1
ali=$2
model=$3
lang=$4

tmpfile=$(mktemp)

echo "%%%%%% .pdf STATS, $tag %%%%%%"
analyze-counts --binary=false --rescale-to-probs=true --show-histogram=true \
  "ark:ali-to-pdf --print-args=false $model \"$ali\" ark:- 2>/dev/null |" \
  $tmpfile.0 2>&1
echo

echo "%%%%%% .phone STATS, $tag %%%%%%"
#prob stats
analyze-counts --binary=false --rescale-to-probs=true --show-histogram=true \
  "ark:ali-to-phones --print-args=false --per-frame=true $model \"$ali\" ark:- |" \
  $tmpfile.1 2>&1
#frame stats
analyze-counts --binary=false \
  "ark:ali-to-phones --print-args=false --per-frame=true $model \"$ali\" ark:- |" \
  $tmpfile.2 2>/dev/null
echo

echo "%%%%%% .ali STATS, $tag %%%%%%"
analyze-counts --binary=false --rescale-to-probs=true --show-histogram=true "$ali" /dev/null 2>&1
echo

echo "%%%%%% .phone STATS (VERBOSE), $tag %%%%%%"
#paste and show the logs
cat $tmpfile.1 | sed -e 's|^\s*\[ ||' -e 's|\]||' | tr ' ' '\n' >$tmpfile.1a
cat $tmpfile.2 | sed -e 's|^\s*\[ ||' -e 's|\]||' | tr ' ' '\n' >$tmpfile.2a
paste $tmpfile.1a $tmpfile.2a > $tmpfile
paste $lang/phones.txt $tmpfile | awk '{printf "%10s %4d  %f %d\n", $1, $2, $3, $4;}' 
echo

echo "%%%%%% .pdf STATS (VERBOSE), $part %%%%%%"
cat $tmpfile.0
echo "%%%%%% END"

rm $tmpfile{,.0,.1,.2,.1a,.2a}



