#!/bin/bash

# Apache 2.0.  Copyright 2013, Hong Kong University of Science and Technology (author: Ricky Chan Ho Yin)

# This script obtains the mixture weights from multiple interpolated LMs produced from build_interpolate_lm_4gram_sri.sh, 
# and gives the average mixture weights. This script can be applied when mixture weights of a target interpolated LM are 
# obtained from K-fold cross validation.

function printUsage {
  echo "Usage: $0 mix_model_name1 mix_model_name2 ... ... mix_model_nameN ug|bg|tg|fg"
  echo "Example: $0 mix_model_name1 mix_model_name2 mix_model_name3 tg"
}

if [ $# -lt 3 ]; then
  printUsage
  exit
fi

args=("$@")

lst=$(( $# - 1 ))
order=${args[${lst}]}

if [ $order != "ug" ] && [ $order != "bg" ] && [ $order != "tg" ] && [ $order != "fg" ]
then
  printUsage  
  exit
fi

i=0
while [ $i -lt $lst ]
do
  if [ ! -e ${args[$i]}/${args[$i]}.$order.wgt ]; then
    echo "Interpolated LM mixture weights ${args[$i]}/${args[$i]}.$order.wgt not found!"
    exit 
  fi
  let i++
done

cmd="grep -h lambda "
i=0;
while [ $i -lt $lst ]
do
  cmd=`echo $cmd ${args[$i]}/${args[$i]}.$order.wgt `
  let i++
done

if [ -f cv_weight ]; then
  echo " original cv_weight file move to cv_weight.bak"
  echo
  mv cv_weight cv_weight.bak
fi
echo " $cmd | cut -d'(' -f 2 | sed 's/)//g' > cv_weight "
echo
`echo $cmd` | cut -d'(' -f 2 | sed 's/)//g' > cv_weight

if [ -f avg_cv_weight ]; then
  echo " original avg_cv_weight file move to avg_cv_weight.bak"
  echo
  mv avg_cv_weight avg_cv_weight.bak
fi
awk '{for(i=1; i<=NF; i++) {array[i] = array[i] + $i} }; END{for(i=1; i<=NF; i++) printf array[i]/NR " "; printf "\n";}'  cv_weight > avg_cv_weight

