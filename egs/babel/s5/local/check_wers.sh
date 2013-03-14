#!/bin/bash


check_wer () {
  dir=$1
  if [ -d $dir ]; then 
    seen_dir=false
    for ddir in $dir/decode*; do
      if [ -d $ddir ]; then
        seen_dir=true
        printf " % -40s " $ddir
        line=`grep Sum $ddir/score_*/*.sys 2>/dev/null | utils/best_wer.sh`
        if [ -z "$line" ]; then echo "------"
        else echo $line | cut -c 1-65; fi
      fi
    done
    ! $seen_dir && echo "$dir ********** no decode dirs"
  fi

}

for dir in exp/tri{2,3,4,5} exp/sgmm5 exp/sgmm5_mmi_b0.1 exp/tri5_nnet exp/tri6_nnet; do
  check_wer $dir
done
