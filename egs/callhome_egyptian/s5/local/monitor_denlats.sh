#!/usr/bin/env bash

currentJob=0

dir=/export/a04/gkumar/kaldi-trunk/egs/fishcall_es/j-matt/exp/sgmm2x_6a_denlats

for f in $dir/.done.*; do
    d=`echo ${f##*/} | awk 'BEGIN {FS="."} {print $3}'`
    if [ $d -gt $currentJob ]; then
        currentJob=$d
    fi
done

currentJob=$((currentJob+1))

echo Currently processing job : $currentJob

for i in $(seq 210); do
    job[$i]=$i
done

dir=/export/a04/gkumar/kaldi-trunk/egs/fishcall_es/j-matt/exp/sgmm2x_6a_denlats/log/$currentJob/q

for f in $dir/done.*; do
    d=`echo ${f##*/} | awk 'BEGIN {FS="."} {print $3}'`
    unset job[$d]
done

echo sub-splits left : ${#job[@]}
echo ${job[@]}
