#!/bin/bash


check_model () {
  model=$1
  if [ -s $model ]; then echo $model
  else 
    dir=`dirname $model`
    latest_model=`ls -lt $dir/{?,??}.mdl 2>/dev/null | head -1 | awk '{print $9}'`
    echo "*$model is not there, latest is: $latest_model"
  fi
}

for model in exp/mono/final.mdl exp/tri{1,2,3}/final.mdl; do
  check_model $model
done

if [ ! -f exp/tri4/final.mdl ]; then
  echo "*exp/tri4/final.mdl is not there*"
  exit 1
fi

if [ -f exp/tri4/trans.1 ]; then # This is LimitedLP.
  models="exp/tri4/final.alimdl exp/sgmm5/final.alimdl exp/sgmm5_mmi_b0.1/final.mdl exp/tri5_nnet/final.mdl"
else
  models="exp/tri4/final.mdl exp/tri5/final.alimdl exp/sgmm5/final.alimdl exp/sgmm5_mmi_b0.1/final.mdl exp/tri6_nnet/final.mdl"
fi
models="$models exp_BNF/tri5/final.mdl exp_BNF/tri6/final.alimdl exp_BNF/sgmm7/final.alimdl"

for model in $models; do
  check_model $model
done


