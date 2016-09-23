#!/bin/bash
#
# This script extract example scp file as egs.*.scp using ranges.*.scp 
# in multilingual training. 

if [ $# -lt 4 ]; then
  echo "$0: Usage: $0 num-langs [<egs-dir-lang-1> .. <egs-dir-lang-n>] <ranges.1> <scp-1>"
  echo " e.g.: $0 2 exp/lang1/egs exp/lang1/egs exp/multi/egs/temp/range.1 exp/multi/egs/egs.1.scp "
  exit 1;
fi

num_langs=$1 # num of languages used in multilingual training
shift
args=("$@")
for l in $(seq 0 $[$num_langs-1]); do
  multi_egs_dirs[$l]=${args[$l]}
done
echo multi_egs_dirs = ${multi_egs_dirs[@]}
range_file=${args[-2]}
scp_file=${args[-1]}

rm -rf $scp_file
while read -r line 
do
  range=($line);
  lang_id=${range[0]}
  start_egs=${range[1]}
  end_egs=$[$start_egs+${range[2]}]
  awk -v s="$start_egs" -v e="$end_egs" 'NR >= s && NR < e' ${multi_egs_dirs[$lang_id]}/egs.scp >> $scp_file; 
done < $range_file 
