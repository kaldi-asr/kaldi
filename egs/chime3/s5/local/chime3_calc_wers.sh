#!/bin/bash

#  Copyright  2015  Mitsubishi Electric Research Laboratories (Author: Shinji Watanabe)
#  Apache 2.0.

set -e

if [ $# -ne 2 ]; then
  printf "\nUSAGE: %s <training experiment directory> <enhancement method>\n\n" `basename $0`
  printf "%s exp/tri3b_tr05_sr_noisy noisy\n\n" `basename $0`
  exit 1;
fi

echo "$0 $@"  # Print the command line for logging

. path.sh

eval_flag=false # make it true when the evaluation data are released

dir=$1
enhan=$2

echo "compute dt05 WER for each location"
echo ""
for a in `find $dir/decode_tgpr_5k_dt05_real_$enhan/ | grep "\/wer_" | awk -F'[/]' '{print $NF}' | sort`; do
    echo -n "$a "
    if [ -e $dir/decode_tgpr_5k_dt05_simu_$enhan ]; then
	cat $dir/decode_tgpr_5k_dt05_{real,simu}_$enhan/$a | grep WER | awk '{err+=$4} {wrd+=$6} END{printf("%.2f\n",err/wrd*100)}'
    else
	cat $dir/decode_tgpr_5k_dt05_real_$enhan/$a | grep WER | awk '{err+=$4} {wrd+=$6} END{printf("%.2f\n",err/wrd*100)}'
    fi
done | sort -n -k 2 | head -n 1 > $dir/log/best_wer_$enhan

lmw=`cut -f 1 -d" " $dir/log/best_wer_$enhan | cut -f 2 -d"_"`
echo "-------------------"
printf "best overall dt05 WER %s" `cut -f 2 -d" " $dir/log/best_wer_$enhan`
echo -n "%"
printf " (language model weight = %s)\n" $lmw
echo "-------------------"
for e_d in dt05 et05; do
  for task in simu real; do
    rdir=$dir/decode_tgpr_5k_${e_d}_${task}_$enhan
    if [ -e $rdir ]; then
      for a in _BUS _CAF _PED _STR; do
	grep $a $rdir/scoring/test_filt.txt \
	  > $rdir/scoring/test_filt_$a.txt
	cat $rdir/scoring/$lmw.tra \
	  | utils/int2sym.pl -f 2- $rdir/../graph_tgpr_5k/words.txt \
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
    fi
  done
done
echo ""

echo "-------------------"
echo "1-best transcription"
echo "-------------------"
for task in simu real; do
    rdir=$dir/decode_tgpr_5k_et05_${task}_$enhan
    cat $rdir/scoring/$lmw.tra \
	| utils/int2sym.pl -f 2- $rdir/../graph_tgpr_5k/words.txt \
	| sed s:\<UNK\>::g
done
