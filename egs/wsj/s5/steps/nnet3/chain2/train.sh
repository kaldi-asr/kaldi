#!/usr/bin/env bash

# Copyright   2019  Johns Hopkins University (Author: Daniel Povey).  Apache 2.0.
# Copyright   2019  Idiap Research Institute (Author: Srikanth Madikeri).  Apache 2.0.


# Begin configuration section
stage=-2
cmd=run.pl
gpu_cmd_opt=
leaky_hmm_coefficient=0.1
xent_regularize=0.1
apply_deriv_weights=false   # you might want to set this to true in unsupervised training
                            # scenarios.
memory_compression_level=2  # Enables us to use larger minibatch size than we
                            # otherwise could, but may not be optimal for speed
                            # (--> set to 0 if you have plenty of memory.
dropout_schedule=
srand=0
max_param_change=2.0    # we use a smaller than normal default (it's normally
                        # 2.0), because there are two models (bottom and top).
use_gpu=yes   # can be "yes", "no", "optional", "wait"
print_interval=10
momentum=0.0
parallel_train_opts=
verbose_opt=

common_opts=           # Options passed through to nnet3-chain-train and nnet3-chain-combine

num_epochs=4.0   #  Note: each epoch may actually contain multiple repetitions of
                 #  the data, for various reasons:
                 #    using the --num-repeats option in process_egs.sh
                 #    data augmentation
                 #    different data shifts (this includes 3 different shifts
                 #    of the data if frame_subsampling_factor=3 (see $dir/init/info.txt)

num_jobs_initial=1
num_jobs_final=1
initial_effective_lrate=0.001
final_effective_lrate=0.0001
minibatch_size=32  # This is how you set the minibatch size. 

max_iters_combine=80
max_models_combine=20
diagnostic_period=5    # Get diagnostics every this-many iterations

shuffle_buffer_size=1000  # This "buffer_size" variable controls randomization of the groups
                          # on each iter.


l2_regularize=
out_of_range_regularize=0.01
multilingual_eg=false

# End configuration section



echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;


if [ $# != 2 ]; then
  echo "Usage: $0  [options] <egs-dir>  <model-dir>"
  echo " e.g.: $0 exp/chain/tdnn1a_sp/egs  exp/chain/tdnn1a_sp"
  echo ""
  echo "This is the default script to train acoustic models for chain2 recipes."
  echo "The script requires two arguments:"
  echo "<egs-dir>: directory where egs files are stored"
  echo "<model-dir>: directory where the final model will be stored"
  echo ""
  echo "See the top of the script to check possible options to pass to it."
  exit 1
fi

egs_dir=$1
dir=$2

set -e -u  # die on failed command or undefined variable

steps/chain2/validate_randomized_egs.sh $egs_dir

for f in $dir/init/info.txt; do
  if [ ! -f $f ]; then
    echo "$0: expected file $f to exist"
    exit 1
  fi
done
cat $egs_dir/info.txt >> $dir/init/info.txt


frame_subsampling_factor=$(awk '/^frame_subsampling_factor/ {print $2}' <$dir/init/info.txt)
num_scp_files=$(awk '/^num_scp_files/ {print $2}' <$egs_dir/info.txt)

if [ $stage -le -2 ]; then
    echo "$0: Generating training schedule"
    steps/chain2/internal/get_train_schedule.py \
      --frame-subsampling-factor=$frame_subsampling_factor \
      --num-jobs-initial=$num_jobs_initial \
      --num-jobs-final=$num_jobs_final \
      --num-epochs=$num_epochs \
      --dropout-schedule="$dropout_schedule" \
      --num-scp-files=$num_scp_files \
      --frame-subsampling-factor=$frame_subsampling_factor \
      --initial-effective-lrate=$initial_effective_lrate \
      --final-effective-lrate=$final_effective_lrate \
      --schedule-out=$dir/schedule.txt
fi


if [ "$use_gpu" != "no" ]; then gpu_cmd_opt="--gpu 1"; else gpu_cmd_opt=""; fi

num_iters=$(wc -l <$dir/schedule.txt)

echo "$0: will train for $num_epochs epochs = $num_iters iterations"

# source the 1st line of schedule.txt in the shell; this sets
# lrate and dropout_opt, among other variables.
. <(head -n 1 $dir/schedule.txt)
langs=$(awk '/^langs/ { $1=""; print; }' <$dir/init/info.txt | tail -1)
num_langs=$(echo $langs | wc -w)

mkdir -p $dir/log

# Copy models with initial learning rate and dropout options from $dir/init to $dir/0
if [ $stage -le -1 ]; then
  echo "$0: Copying transition model"
  if [ $num_langs -eq 1 ]; then
      echo "$0: Num langs is 1"
      cp $dir/init/default.raw $dir/0.raw
      if [ -f $dir/init/default_trans.mdl ]; then
          cp $dir/init/default_trans.mdl $dir/0_trans.mdl 
      fi
  else
      echo "$0: Num langs is $num_langs"
      cp $dir/init/multi.raw $dir/0.raw
  fi
fi


l2_regularize_opt=""
if [ ! -z $l2_regularize ]; then
    l2_regularize_opt="--l2-regularize=$l2_regularize"
fi

x=0
if [ $stage -gt $x ]; then x=$stage; fi

[ $max_models_combine -gt $[num_iters/2] ] && max_models_combine=$[num_iters/2];
combine_start_iter=$[num_iters+1-max_models_combine]

while [ $x -lt $num_iters ]; do
  # Source some variables fromm schedule.txt.  The effect will be something
  # like the following:
  # iter=0; num_jobs=2; inv_num_jobs=0.5; scp_indexes=(pad 1 2); frame_shifts=(pad 1 2); dropout_opt="--edits='set-dropout-proportion name=* proportion=0.0'" lrate=0.002
  . <(grep "^iter=$x;" $dir/schedule.txt)

  echo "$0: training, iteration $x of $num_iters, num-jobs is $num_jobs"

  next_x=$[$x+1]
  den_fst_dir=$egs_dir/misc
  model_out_prefix=$dir/${next_x}
  model_out=${model_out_prefix}.mdl
  multilingual_eg_opts=
  if $multilingual_eg; then
       multilingual_eg_opts="--multilingual-eg=true"
  fi

  # for the first 4 iterations, plus every $diagnostic_period iterations, launch
  # some diagnostic processes.  We don't do this on iteration 0, because
  # the batchnorm stats wouldn't be ready
  if [ $x -gt 0 ] && [ $[x%diagnostic_period] -eq 0 -o $x -lt 5 ]; then

    [ -f $dir/.error_diagnostic ] && rm $dir/.error_diagnostic
    for name in train heldout; do
      egs_opts=
      if $multilingual_eg; then
          weight_rspecifier=$egs_dir/diagnostic_${name}.weight.ark
          [[ -f $weight_rspecifier ]] && egs_opts="--weights=ark:$weight_rspecifier"
      fi
      $cmd $gpu_cmd_opt $dir/log/diagnostic_${name}.$x.log \
         nnet3-chain-train2 --use-gpu=$use_gpu \
            --leaky-hmm-coefficient=$leaky_hmm_coefficient \
            --xent-regularize=$xent_regularize \
            --out-of-range-regularize=$out_of_range_regularize \
            $l2_regularize_opt \
            --print-interval=10  \
           "nnet3-copy --learning-rate=$lrate $dir/${x}.raw - |" $den_fst_dir \
           "ark:nnet3-chain-copy-egs $egs_opts scp:$egs_dir/${name}_subset.scp ark:- | nnet3-chain-merge-egs $multilingual_eg_opts --minibatch-size=1:64 ark:- ark:-|" \
           $dir/${next_x}_${name}.mdl || touch $dir/.error_diagnostic &

       # Make sure we do not run more than $num_jobs_final at once
       [ $num_jobs_final -eq 1 ] && wait

    done
    wait
  fi

  if [ $x -gt 0 ]; then
    # This doesn't use the egs, it only shows the relative change in model parameters.
    $cmd $dir/log/progress.$x.log \
      nnet3-show-progress --use-gpu=no $dir/$(($x-1)).raw $dir/${x}.raw '&&' \
        nnet3-info $dir/${x}.raw &
  fi

  cache_io_opt="--write-cache=$dir/cache.$next_x"
  if [ $x -gt 0 -a -f $dir/cache.$x ]; then
      cache_io_opt="$cache_io_opt --read-cache=$dir/cache.$x"
  fi
  for j in $(seq $num_jobs); do
    scp_index=${scp_indexes[$j]}
    frame_shift=${frame_shifts[$j]}

    egs_opts=
    if $multilingual_eg; then
        weight_rspecifier=$egs_dir/train.weight.$scp_index.ark
        [[ -f $weight_rspecifier ]] && egs_opts="--weights=ark:$weight_rspecifier"
    fi
    $cmd $gpu_cmd_opt $dir/log/train.$x.$j.log \
         nnet3-chain-train2  \
             $parallel_train_opts $verbose_opt \
             --out-of-range-regularize=$out_of_range_regularize \
             $cache_io_opt \
             --use-gpu=$use_gpu --apply-deriv-weights=$apply_deriv_weights \
             --leaky-hmm-coefficient=$leaky_hmm_coefficient --xent-regularize=$xent_regularize \
             --print-interval=$print_interval --max-param-change=$max_param_change \
             --momentum=$momentum \
             --l2-regularize-factor=$inv_num_jobs \
             $l2_regularize_opt \
             --srand=$srand \
             "nnet3-copy --learning-rate=$lrate $dir/${x}.raw - |" $den_fst_dir \
             "ark:nnet3-chain-copy-egs $egs_opts --frame-shift=$frame_shift scp:$egs_dir/train.$scp_index.scp ark:- | nnet3-chain-shuffle-egs --buffer-size=$shuffle_buffer_size --srand=$x ark:- ark:- | nnet3-chain-merge-egs $multilingual_eg_opts --minibatch-size=$minibatch_size ark:- ark:-|" \
             ${model_out_prefix}.$j.raw || touch $dir/.error &
  done
  wait
  if [ -f $dir/.error ]; then
    echo "$0: error detected training on iteration $x"
    exit 1
  fi
  if [ $x -ge 1 ]; then
      models_to_average=$(for j in `seq $num_jobs`; do echo ${model_out_prefix}.$j.raw; done)
      $cmd $dir/log/average.$x.log \
          nnet3-average $models_to_average $dir/$next_x.raw  || exit 1;
      rm $models_to_average
  else
      lang=$(echo $langs | awk '{print $1}')
      model_index=`steps/nnet3/chain2/internal/get_best_model.sh --output output-${lang} $dir/log/train.$x.*.log`
      cp ${model_out_prefix}.$model_index.raw $dir/$next_x.raw
      rm ${model_out_prefix}.*.raw
  fi
  [ -f $dir/$x/.error_diagnostic ] && echo "$0: error getting diagnostics on iter $x" && exit 1;

  if [ -f $dir/cache.$x ]; then
      rm $dir/cache.$x
  fi
  delete_iter=$[x-2]
  if [ $delete_iter -lt $combine_start_iter ]; then
      if [ -f $dir/$delete_iter.raw ]; then
          rm $dir/$delete_iter.raw
      fi
  fi
  if [ -f $dir/${next_x}_train.mdl ]; then
      rm $dir/${next_x}_{train,heldout}.mdl
  fi
  x=$[x+1]
done



if [ $stage -le $num_iters ]; then
  echo "$0: doing model combination"
  den_fst_dir=$egs_dir/misc
  input_models=$(for x in $(seq $combine_start_iter $num_iters); do echo $dir/${x}.raw; done)
  output_model_dir=$dir/final

   $cmd $gpu_cmd_opt $dir/log/combine.log \
      nnet3-chain-combine2 --use-gpu=$use_gpu \
        --leaky-hmm-coefficient=$leaky_hmm_coefficient \
        --print-interval=10  \
        $den_fst_dir $input_models \
        "ark:nnet3-chain-merge-egs $multilingual_eg_opts  scp:$egs_dir/train_subset.scp ark:-|" \
        $dir/final.raw || exit 1;
   if ! $multilingual_eg; then
       nnet3-copy  --edits="rename-node old-name=output new-name=output-dummy; rename-node old-name=output-default new-name=output" \
          $dir/final.raw - | \
          nnet3-am-init $dir/0_trans.mdl - $dir/final.mdl
   fi

   # Compute the probability of the final, combined model with
   # the same subset we used for the previous diagnostic processes, as the
   # different subsets will lead to different probs.
   [ -f $dir/.error_diagnostic ] && rm $dir/.error_diagnostic
   for name in train heldout; do
     egs_opts=
     if $multilingual_eg; then
       weight_rspecifier=$egs_dir/diagnostic_${name}.weight.ark
       [[ -f $weight_rspecifier ]] && egs_opts="--weights=ark:$weight_rspecifier"
     fi
     $cmd $gpu_cmd_opt $dir/log/diagnostic_${name}.final.log \
       nnet3-chain-train2 --use-gpu=$use_gpu \
         --leaky-hmm-coefficient=$leaky_hmm_coefficient \
         --xent-regularize=$xent_regularize \
         --out-of-range-regularize=$out_of_range_regularize \
         $l2_regularize_opt \
         --print-interval=10  \
         $dir/final.raw  $den_fst_dir \
         "ark:nnet3-chain-copy-egs $egs_opts scp:$egs_dir/${name}_subset.scp ark:- | nnet3-chain-merge-egs $multilingual_eg_opts --minibatch-size=1:64 ark:- ark:-|" \
         $dir/final_${name}.mdl || touch $dir/.error_diagnostic &
   done

   if [ -f $dir/final_train.mdl ]; then
     rm $dir/final_{train,heldout}.mdl
   fi
fi

if [[ ! $multilingual_eg ]] && [[ ! -f $dir/final.mdl ]]; then
  echo "$0: $dir/final.mdl does not exist."
  # we don't want to clean up if the training didn't succeed.
  exit 1;
fi

sleep 2

echo "$0: done"

steps/info/chain_dir_info.pl $dir

exit 0
