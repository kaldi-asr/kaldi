#!/bin/bash

train_text=rnnlm/train.txt
dev_text=rnnlm/dev.txt
wordlist_in=rnnlm/wordlist.in
wordlist_out=rnnlm/wordlist.out

history_length=20

echo "$0 $@"  # Print the command line for logging                              
                                                                                
[ -f ./path.sh ] && . ./path.sh; # source the path.                             
. parse_options.sh || exit 1;

output=$1

for i in $train_text $dev_text $wordlist_in $wordlist_out; do
  [ -f $i ] || echo "$i file not found" || exit 1;
done

rnnlm-get-egs --history=$history_length --binary=false $train_text $dev_text $wordlist_in $wordlist_out $output
