#!/bin/bash

# Copyright 2016 MERL (author: Shinji Watanabe)

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

. ./cmd.sh
. ./path.sh

lmw=15
am="tri2a"
lm="bg_5k"
decode=""

. utils/parse_options.sh

if [ ! -z $decode ]; then
  decode="_$decode"
fi

dir="exp/$am/decode${decode}_${lm}_REVERB_"
echo "####################"
echo "${dir}*dt*"
for a in `echo ${dir}*dt* | tr " " "\n" | grep -v "A\.si"`; do
  echo $a | awk -F '_' '{for(i=NF-6;i<NF;i++){printf("%s%s",$i,OFS="_")}print $NF}' | tr '\n' '\t'
  grep WER $a/wer_${lmw} | awk '{print $2}'
done | tee exp/$am/decode_${decode}_${lm}_dt.log
echo -n -e "Avg_Real(`cat exp/$am/decode_${decode}_${lm}_dt.log | grep RealData | wc -l`)\t"
cat exp/$am/decode_${decode}_${lm}_dt.log | grep RealData | awk '{m+=$2} END{printf("%5.2f\n", m/NR);}'
echo -n -e "Avg_Sim(`cat exp/$am/decode_${decode}_${lm}_dt.log | grep SimData | wc -l`)\t"
cat exp/$am/decode_${decode}_${lm}_dt.log | grep SimData | awk '{m+=$2} END{printf("%5.2f\n", m/NR);}'
echo ""

echo "${dir}*et*"
for a in `echo ${dir}*et* | tr " " "\n" | grep -v "A\.si"`; do
  echo $a | awk -F '_' '{for(i=NF-6;i<NF;i++){printf("%s%s",$i,OFS="_")}print $NF}' | tr '\n' '\t'
  grep WER $a/wer_${lmw} | awk '{print $2}'
done | tee exp/$am/decode_${decode}_${lm}_et.log
echo -n -e "Avg_Real(`cat exp/$am/decode_${decode}_${lm}_et.log | grep RealData | wc -l`)\t"
cat exp/$am/decode_${decode}_${lm}_et.log | grep RealData | awk '{m+=$2} END{printf("%5.2f\n", m/NR);}'
echo -n -e "Avg_Sim(`cat exp/$am/decode_${decode}_${lm}_et.log | grep SimData | wc -l`)\t"
cat exp/$am/decode_${decode}_${lm}_et.log | grep SimData | awk '{m+=$2} END{printf("%5.2f\n", m/NR);}'

echo ""
