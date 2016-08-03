#!/bin/bash

#  Copyright  2015  Mitsubishi Electric Research Laboratories (Author: Shinji Watanabe)
#  Apache 2.0.

set -e

if [ $# -ne 3 ]; then
  printf "\nUSAGE: %s <training experiment directory> <enhancement method> <graph_dir>\n\n" `basename $0`
  printf "%s exp/tri3b_tr05_sr_noisy noisy exp/tri4a_dnn_tr05_sr_noisy/graph_tgpr_5k\n\n" `basename $0`
  exit 1;
fi

echo "$0 $@"  # Print the command line for logging

. path.sh

eval_flag=false # make it true when the evaluation data are released

dir=$1
enhan=$2
graph_dir=$3

echo "compute WER for each location"
echo ""
# collect scores
for x in `find $dir/ -type d -name "*_it*" | awk -F "_it" '{print $NF}' | sort | uniq`; do
    for y in `find $dir/*_${enhan}_it*/ | grep "\/wer_" | awk -F'[/]' '{print $NF}' | sort | uniq`; do
	echo -n "${x}_$y "
	cat $dir/decode_tgpr_5k_dt05_{real,simu}_${enhan}_it$x/$y | grep WER | awk '{err+=$4} {wrd+=$6} END{printf("%.2f\n",err/wrd*100)}'
    done
done | sort -n -k 2 | head -n 1 > $dir/log/best_wer_$enhan

lmw=`cut -f 1 -d" " $dir/log/best_wer_$enhan | awk -F'[_]' '{print $NF}'`
it=`cut -f 1 -d" " $dir/log/best_wer_$enhan | awk -F'[_]' '{print $1}'`
echo "-------------------"
printf "best overall dt05 WER %s" `cut -f 2 -d" " $dir/log/best_wer_$enhan`
echo -n "%"
printf " (language model weight = %s)\n" $lmw
printf " (Number of iterations = %s)\n" $it
echo "-------------------"
for e_d in dt05 et05; do
  for task in simu real; do
    rdir=$dir/decode_tgpr_5k_${e_d}_${task}_${enhan}_it$it
    for a in _BUS _CAF _PED _STR; do
      grep $a $rdir/scoring/test_filt.txt \
	> $rdir/scoring/test_filt_$a.txt
      cat $rdir/scoring/$lmw.tra \
	| utils/int2sym.pl -f 2- $graph_dir/words.txt \
	| sed s:\<UNK\>::g \
	| compute-wer --text --mode=present ark:$rdir/scoring/test_filt_$a.txt ark,p:- \
	1> $rdir/${a}_wer_$lmw 2> /dev/null
    done
    echo -n "${e_d}_${task} WER: `grep WER $rdir/wer_$lmw | cut -f 2 -d" "`% (Average), "
    echo -n "`grep WER $rdir/_BUS_wer_$lmw | cut -f 2 -d" "`% (BUS), "
    echo -n "`grep WER $rdir/_CAF_wer_$lmw | cut -f 2 -d" "`% (CAFE), "
    echo -n "`grep WER $rdir/_PED_wer_$lmw | cut -f 2 -d" "`% (PEDESTRIAN), "
    echo -n "`grep WER $rdir/_STR_wer_$lmw | cut -f 2 -d" "`% (STREET)"
    echo ""
    echo "-------------------"
  done
done

echo "-------------------"
echo "1-best transcription"
echo "-------------------"
for task in simu real; do
    rdir=$dir/decode_tgpr_5k_et05_${task}_${enhan}_it$it
    cat $rdir/scoring/$lmw.tra \
	| utils/int2sym.pl -f 2- $graph_dir/words.txt \
	| sed s:\<UNK\>::g
done
