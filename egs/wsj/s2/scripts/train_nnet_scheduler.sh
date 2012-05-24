#!/bin/bash

##############################
#check for obligatory parameters
echo
echo %%% CONFIG
echo TRAIN_TOOL: ${TRAIN_TOOL?$0: TRAIN_TOOL not specified}
echo feats_cv: ${feats_cv?$0: feats_cv not specified}
echo feats_tr: ${feats_tr?$0: feats_tr not specified}
echo labels: ${labels?$0: labels not specified}
echo mlp_init: ${mlp_init?$0: mlp_init not specified}
echo max_iters: ${max_iters?$0: max_iters not specified}
echo start_halving_inc: ${start_halving_inc?$0: start_halving_inc not specified}
echo end_halving_inc: ${end_halving_inc?$0: end_halving_inc not specified}
echo halving_factor: ${halving_factor?$0: halving_factor not specified}
echo lrate: ${lrate?$0: lrate not specified}
echo ${l2penalty:+l2penalty: $l2penalty}
echo %%% CONFIG
echo

#template
#echo A: ${A?$0: A not specified}

##############################
#define implicit configuration
#echo bunchsize: ${bunchsize:=512}
#echo cachesize: ${cachesize:=$((64*bunchsize))

#template
#echo A: ${A:=val}



##############################
#start training

#prerun cross-validation
$TRAIN_TOOL --cross-validate=true $mlp_init "$feats_cv" "$labels" &> $dir/log/prerun.log || { cat $dir/log/prerun.log; exit 1; }
acc=$(cat $dir/log/prerun.log | awk '/FRAME_ACCURACY/{ acc=$3; sub(/%/,"",acc); } END{print acc}')
echo "CROSSVAL PRERUN ACCURACY $acc"

#training
mlp_best=$mlp_init
mlp_base=${mlp_init##*/}; mlp_base=${mlp_base%.*}

iter=0
halving=0
for iter in $(seq -w $max_iters); do
  echo -n "ITERATION $iter: "
  mlp_next=$dir/nnet/${mlp_base}_iter${iter}
  
  #training
  $TRAIN_TOOL --learn-rate=$lrate ${l2penalty:+ --l2-penalty=$l2penalty} $mlp_best "$feats_tr" "$labels" $mlp_next &> $dir/log/iter$iter.log || exit 1; 
  tr_acc=$(cat $dir/log/iter$iter.log | awk '/FRAME_ACCURACY/{ acc=$3; sub(/%/,"",acc); } END{print acc}')
  echo -n "TRAIN ACCURACY $(printf "%.2f" $tr_acc) LRATE $(printf "%.6g" $lrate), "
  
  #cross-validation
  $TRAIN_TOOL --cross-validate=true $mlp_next "$feats_cv" "$labels" 1>>$dir/log/iter$iter.log 2>>$dir/log/iter$iter.log || exit 1;
  acc_new=$(cat $dir/log/iter$iter.log | awk '/FRAME_ACCURACY/{ acc=$3; sub(/%/,"",acc); } END{print acc}')
  echo -n "CROSSVAL ACCURACY $(printf "%.2f" $acc_new), "

  #accept or reject new parameters
  acc_prev=$acc
  if [ "1" == "$(awk "BEGIN{print($acc_new>$acc);}")" ]; then
    acc=$acc_new
    mlp_best=$dir/nnet/${mlp_base}_iter${iter}_lrate${lrate}_tr$(printf "%.2f" $tr_acc)_cv$(printf "%.2f" $acc_new)
    mv $mlp_next $mlp_best
    echo nnet $mlp_best accepted
  else
    mlp_reject=$dir/nnet/${mlp_base}_iter${iter}_lrate${lrate}_tr$(printf "%.2f" $tr_acc)_cv$(printf "%.2f" $acc_new)_rejected
    mv $mlp_next $mlp_reject
    echo nnet $mlp_reject rejected 
  fi

  #stopping criterion
  if [[ "1" == "$halving" && "1" == "$(awk "BEGIN{print($acc < $acc_prev+$end_halving_inc)}")" ]]; then
    echo finished, too small improvement $(awk "BEGIN{print($acc-$acc_prev)}")
    break
  fi

  #start annealing when improvement is low
  if [ "1" == "$(awk "BEGIN{print($acc < $acc_prev+$start_halving_inc)}")" ]; then
    halving=1
  fi
  
  #do annealing
  if [ "1" == "$halving" ]; then
    lrate=$(awk "BEGIN{print($lrate*$halving_factor)}")
  fi
done

#select the best network
if [ $mlp_best != $mlp_init ]; then 
  mlp_final=${mlp_best}_final_
  cp $mlp_best $mlp_final
fi


