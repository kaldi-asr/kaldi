#!/bin/bash

# this script is used for comparing trained models between systems.
# e.g. local/nnet3/compare.sh exp/resnet1{b,c}


if [ $# == 0 ]; then
  echo "Usage: $0: <dir1> [<dir2> ... ]"
  echo "e.g.: $0 exp/resnet1{b,c}"
  exit 1
fi

echo "# $0 $*"



echo -n "# System               "
for x in $*; do   printf "% 12s" " $(basename $x)";   done
echo


echo -n "# final test accuracy: "
for x in $*; do
  acc=$(grep acc $x/log/compute_prob_valid.final.log | awk '{print $8}')
  printf "% 12s" $acc
done

echo
echo -n "# final train accuracy: "
for x in $*; do
  acc=$(grep acc $x/log/compute_prob_train.final.log | awk '{print $8}')
  printf "% 12s" $acc
done

echo
echo -n "# final test objf:      "
for x in $*; do
  objf=$(grep log-like $x/log/compute_prob_valid.final.log | awk '{print $8}')
  printf "% 12s" $objf
done

echo
echo -n "# final train objf:     "
for x in $*; do
  objf=$(grep log-like $x/log/compute_prob_train.final.log | awk '{print $8}')
  printf "% 12s" $objf
done

echo
echo -n "# num-parameters:      "
for x in $*; do
  params=$(grep num-parameters $x/log/progress.1.log | awk '{print $2}')
  printf "% 12s" $params
done

echo
