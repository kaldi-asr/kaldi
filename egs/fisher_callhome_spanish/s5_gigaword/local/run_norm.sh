#!/bin/bash

set -euo pipefail

punctuation_symbols=( "," "\"" "\`" "\:" "(" ")" "-" ";" "?" "!" "/" "_" "{" "}" "*" )

config=$1
path_prefix=$2
data=$3
job=$4
dir=$5

substitute_arg=""
num_syms=0

for i in "${punctuation_symbols[@]}"; do
    symbol=${punctuation_symbols[${num_syms}]}
    if [ $num_syms -eq 0 ]; then
	substitute_arg="sed 's:${i}: :g'"
    else
	substitute_arg=$substitute_arg" | sed 's:${i}: :g'"
    fi
    substitute_arg=$substitute_arg" |sed 's:${i}$: :g' | sed 's:^${i}: :g'"
    num_syms=$((num_syms+1))
done
mkdir -p $dir/normalize/$job
local/clean_abbrevs_text.py $data/$job $data/"$job"_processed
mv $data/"$job"_processed $data/$job
echo "cat $data/$job | $substitute_arg" > $dir/normalize/$job/substitute.sh
 
bash $dir/normalize/$job/substitute.sh | \
    sed "s: 's:'s:g" | sed "s: 'm:'m:g" | \
    sed "s: \s*: :g" | tr 'A-ZÂÁÀÄÊÉÈËÏÍÎÖÓÔÖÚÙÛÑÇ' 'a-zâáàäêéèëïíîöóôöúùûñç'  > $dir/normalize/$job/text
normalizer_main --config=$config --path_prefix=$path_prefix <$dir/normalize/$job/text >$dir/$job.txt

exit 0;
