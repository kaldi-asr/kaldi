#!/usr/bin/env bash



check_wer () {
  dir=$1
  if [ -d $dir ]; then 
    seen_dir=false
    for ddir in $dir/decode*; do
      if [ -d $ddir ]; then
        seen_dir=true
        printf " % -40s " $ddir
        line=`grep Sum $ddir/score_*/*.sys 2>/dev/null | $char_command | utils/best_wer.sh`
        if [ -z "$line" ]; then echo "------"
        else echo $line | cut -c 1-65; fi
      fi
    done
    ! $seen_dir && echo "$dir ********** no decode dirs"
  fi

}

final=false
char_command="grep -v char"

for n in `seq 10`; do
  if [ "$1" == "--final" ]; then
    final=true
    shift
  fi
  if [ "$1" == "--char" ]; then
    char_command="grep char"
    shift
  fi
done

if [ $# != 0 ]; then 
  echo "Usage: local/check_wers.sh [--final] [--char]"
  exit 1;
fi

if $final; then
  for dir in exp/sgmm5_mmi_b0.1 exp/tri5_nnet exp/tri6_nnet exp_BNF/sgmm7 exp_BNF/sgmm7_mmi_b0.1 exp/combine*; do
    check_wer $dir
  done
else
  for dir in exp/tri{2,3,4,5} exp/sgmm5 exp/sgmm5_mmi_b0.1 exp/tri5_nnet exp/tri6_nnet exp_BNF/* exp/combine_*; do
    check_wer $dir
  done
fi
