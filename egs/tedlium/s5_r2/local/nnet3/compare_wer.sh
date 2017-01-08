#!/bin/bash


echo $0 $*

include_looped=false
if [ "$1" == "--looped" ]; then
  include_looped=true
  shift
fi

echo -n "System               "
for x in $*; do   printf "% 10s" " $(basename $x)";   done
echo

dirnames=(dev dev_rescore test test_rescore)
strings=("WER on dev(orig)     " "WER on dev(rescored) " "WER on test(orig)    " "WER on test(rescored)")

for n in 0 1 2 3; do
   echo -n "${strings[$n]}"
   for x in $*; do
     wer=$(grep Sum $x/decode_${dirnames[$n]}/score*/*ys | utils/best_wer.sh | awk '{print $2}')
     printf "% 10s" $wer
   done
   echo
   if $include_looped; then
     echo -n "        [looped:]    "
     for x in $*; do
       wer=$(grep Sum $x/decode_looped_${dirnames[$n]}/score*/*ys | utils/best_wer.sh | awk '{print $2}')
       printf "% 10s" $wer
     done
     echo
   fi
done


echo -n "Final train prob     "
for x in $*; do
  prob=$(grep Overall $x/log/compute_prob_train.final.log | awk '{printf("%.4f", $8)}')
  printf "% 10s" $prob
done
echo

echo -n "Final valid prob     "
for x in $*; do
  prob=$(grep Overall $x/log/compute_prob_valid.final.log | awk '{printf("%.4f", $8)}')
  printf "% 10s" $prob
done
echo

echo
