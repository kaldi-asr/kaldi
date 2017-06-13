#!/usr/bin/env bash

# To run this, cd to the top level of the repo and type
# misc/maintenance/fix_include_guards.sh

set -e

cd src
rm -rf tmp

for x in */*.h ; do
  name=`echo $x | tr '[a-z]/.-' '[A-Z]___' `
  m=KALDI_${name}_
  n=`grep ifndef $x | awk '{print $2}' | head -n 1`
  if [ "$m" != "$n" ]; then
    echo "$m != $n";
    if [ ! -z "$n" ]; then
      cp $x tmp; sed s/$n/$m/ <tmp >$x;
    else
      echo "Something wrong for file $x, maybe no include guard."
    fi
  fi
done


for x in */*.h ; do 
  name=`echo $x | tr '[a-z]/.-' '[A-Z]___' `
  m=KALDI_${name}_  
  n=`grep endif $x | grep _H_ | sed s://:: | awk '{print $2}' | head -n 1`
  if [ ! -s $n ] && [ "$m" != "$n" ]; then 
    echo "#endif: $m != $n";
    cp $x tmp; sed s/$n/$m/ <tmp > $x;
  fi
done
