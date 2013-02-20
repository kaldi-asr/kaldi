#!/bin/bash

##############################
#check for obligatory parameters
echo
echo %%% CONFIG
echo learn_rate: ${learn_rate?$0: learn_rate not specified}
echo momentum:  ${momentum?$0: momentum not specified}
echo l1_penalty: ${l1_penalty?$0: l1_penalty not specified}
echo l2_penalty: ${l2_penalty?$0: l2_penalty not specified}
echo 
echo bunch_size: ${bunch_size?$0: bunch_size not specified}
echo cache_size: ${cache_size?$0: cache_size not specified}
echo randomize: ${randomize?$0: randomize not specified}
echo
echo max_iters: ${max_iters?$0: max_iters not specified}
echo start_halving_inc: ${start_halving_inc?$0: start_halving_inc not specified}
echo end_halving_inc: ${end_halving_inc?$0: end_halving_inc not specified}
echo halving_factor: ${halving_factor?$0: halving_factor not specified}
echo
echo TRAIN_TOOL: ${TRAIN_TOOL?$0: TRAIN_TOOL not specified}
echo
echo feats_cv: ${feats_cv?$0: feats_cv not specified}
echo feats_tr: ${feats_tr?$0: feats_tr not specified}
echo labels: ${labels?$0: labels not specified}
echo mlp_init: ${mlp_init?$0: mlp_init not specified}
echo ${feature_transform:+feature_transform: $feature_transform}
echo ${min_iters:+min_iters: $min_iters}
echo ${use_gpu_id:+use_gpu_id: $use_gpu_id}
echo %%% CONFIG
echo


##############################
#start training

#choose mlp to start with
mlp_best=$mlp_init
mlp_base=${mlp_init##*/}; mlp_base=${mlp_base%.*}
#optionally resume training from the best epoch
[ -e $dir/mlp_best ] && mlp_best=$(cat $dir/mlp_best)
[ -e $dir/learn_rate ] && learn_rate=$(cat $dir/learn_rate)

#prerun cross-validation
$TRAIN_TOOL --cross-validate=true \
 --bunchsize=$bunch_size --cachesize=$cache_size \
 ${feature_transform:+ --feature-transform=$feature_transform} \
 ${use_gpu_id:+ --use-gpu-id=$use_gpu_id} \
 $mlp_best "$feats_cv" "$labels" \
 2> $dir/log/prerun.log || exit 1;

acc=$(cat $dir/log/prerun.log | awk '/FRAME_ACCURACY/{ acc=$3; sub(/%/,"",acc); } END{print acc}')
echo "CROSSVAL PRERUN ACCURACY $acc"

#resume lr-halving
halving=0
[ -e $dir/halving ] && halving=$(cat $dir/halving)
#training
for iter in $(seq -w $max_iters); do
  echo -n "ITERATION $iter: "
  mlp_next=$dir/nnet/${mlp_base}_iter${iter}
  
  #skip iteration if already done
  [ -e $dir/log/iter$iter.log__DONE ] && echo -n "skipping... " && ls $mlp_next* && continue 
  
  #training
  $TRAIN_TOOL \
   --learn-rate=$learn_rate --momentum=$momentum --l1-penalty=$l1_penalty --l2-penalty=$l2_penalty \
   --bunchsize=$bunch_size --cachesize=$cache_size --randomize=$randomize \
   ${feature_transform:+ --feature-transform=$feature_transform} \
   ${use_gpu_id:+ --use-gpu-id=$use_gpu_id} \
   ${seed:+ --seed=$seed} \
   $mlp_best "$feats_tr" "$labels" $mlp_next \
   2> $dir/log/iter$iter.log || exit 1; 

  tr_acc=$(cat $dir/log/iter$iter.log | awk '/FRAME_ACCURACY/{ acc=$3; sub(/%/,"",acc); } END{print acc}')
  echo -n "TRAIN ACCURACY $(printf "%.2f" $tr_acc) LRATE $(printf "%.6g" $learn_rate), "
  
  #cross-validation
  $TRAIN_TOOL --cross-validate=true \
   --bunchsize=$bunch_size --cachesize=$cache_size \
   ${feature_transform:+ --feature-transform=$feature_transform} \
   ${use_gpu_id:+ --use-gpu-id=$use_gpu_id} \
   $mlp_next "$feats_cv" "$labels" \
   2>>$dir/log/iter$iter.log || exit 1;
  
  acc_new=$(cat $dir/log/iter$iter.log | awk '/FRAME_ACCURACY/{ acc=$3; sub(/%/,"",acc); } END{print acc}')
  echo -n "CROSSVAL ACCURACY $(printf "%.2f" $acc_new), "

  #accept or reject new parameters
  acc_prev=$acc
  if [ "1" == "$(awk "BEGIN{print($acc_new>$acc);}")" ]; then
    acc=$acc_new
    mlp_best=$dir/nnet/${mlp_base}_iter${iter}_learnrate${learn_rate}_tr$(printf "%.2f" $tr_acc)_cv$(printf "%.2f" $acc_new)
    mv $mlp_next $mlp_best
    echo "nnet accepted ($(basename $mlp_best))"
    echo $mlp_best > $dir/mlp_best 
  else
    mlp_reject=$dir/nnet/${mlp_base}_iter${iter}_learnrate${learn_rate}_tr$(printf "%.2f" $tr_acc)_cv$(printf "%.2f" $acc_new)_rejected
    mv $mlp_next $mlp_reject
    echo "nnet rejected ($(basename $mlp_reject))"
  fi

  #rename the log file, so we know that the epoch is over
  mv $dir/log/iter$iter.log $dir/log/iter$iter.log__DONE

  #stopping criterion
  if [[ "1" == "$halving" && "1" == "$(awk "BEGIN{print($acc < $acc_prev+$end_halving_inc)}")" ]]; then
    if [[ "$min_iters" != "" ]]; then
      if [ $min_iters -gt $iter ]; then
        echo we were supposed to finish, but we continue, min_iters : $min_iters
        continue
      fi
    fi
    echo finished, too small improvement $(awk "BEGIN{print($acc-$acc_prev)}")
    break
  fi

  #start annealing when improvement is low
  if [ "1" == "$(awk "BEGIN{print($acc < $acc_prev+$start_halving_inc)}")" ]; then
    halving=1
    echo $halving >$dir/halving
  fi
  
  #do annealing
  if [ "1" == "$halving" ]; then
    learn_rate=$(awk "BEGIN{print($learn_rate*$halving_factor)}")
    echo $learn_rate >$dir/learn_rate
  fi
done

#select the best network
if [ $mlp_best != $mlp_init ]; then 
  mlp_final=${mlp_best}_final_
  ( cd $dir/nnet; ln -s $(basename $mlp_best) $(basename $mlp_final); )
fi


