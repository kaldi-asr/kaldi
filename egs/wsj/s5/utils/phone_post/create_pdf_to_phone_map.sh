#!/bin/bash

set -euo pipefail

if [ $# != 3 ]; then
  echo "Usage: $0 $lang $modeldir $outdir"
  exit 1;
fi

lang=$1
model=$2
outdir=$3

logdir=$outdir/log; mkdir -p $logdir


# Create pseudo_phones.txt
utils/phone_post/create_pseudo_phones.py $lang/phones/roots.txt 2>$logdir/create_pseudo_phones.log >$outdir/pseudo_phones.txt || exit 1;

# Create pdf_to_pseudo_phones.txt
#model=$modeldir/final.mdl
show-transitions $lang/phones.txt "$model" 2>$logdir/show_transitions.log >$outdir/show_transitions.txt || exit 1;

utils/phone_post/show_transitions_to_sym2int.py $outdir/show_transitions.txt $lang/phones/roots.txt | utils/sym2int.pl -f 2 $outdir/pseudo_phones.txt | sort -n -k 1 | uniq > $outdir/pdf_to_pseudo_phone.txt || exit 1;

# Check if pdf-id's are uniq
max=`tail -1 $outdir/pdf_to_pseudo_phone.txt | awk '{print $1;}'`
num_lines=`cat $outdir/pdf_to_pseudo_phone.txt | wc -l`
if [[ $max -ne $num_lines-1 ]]; then
  echo "pdf-id's are not uniq.";
  exit 1;
fi

