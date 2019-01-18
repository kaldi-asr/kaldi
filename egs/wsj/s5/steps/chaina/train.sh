#!/bin/bash

# Copyright   2019  Johns Hopkins University (Author: Daniel Povey).  Apache 2.0.


# Begin configuration section
stage=0
leaky_hmm_coefficient=0.1
xent_regularize=0.1
apply_deriv_weights=false   # you might want to set this to true in unsupervised training
                            # scenarios.
memory_compression_level=2  # Enables us to use larger minibatch size than we
                            # otherwise could, but may not be optimal for speed
                            # (--> set to 0 if you have plenty of memory.
dropout_schedule=
srand=0
max_param_change=1.0    # we use a smaller than normal default (it's normally
                        # 2.0), because there are two models (bottom and top).
use_gpu=yes   # can be "yes", "no", "optional", "wait"

common_opts=           # Options passed through to nnet3-chaina-train and nnet3-chaina-combine

unadapted_top_weight=0.5
unadapted_bottom_weight=0.5

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
groups_per_minibatch=32  # This is how you set the minibatch size.  Note: if
                         # chunks_per_group=4, this would mean 128 chunks per
                         # minibatch.

max_iters_combine=80
max_models_combine=20
diagnostic_period=5    # Get diagnostics every this-many iterations

shuffle_buffer_size=1000  # This "buffer_size" variable controls randomization of the groups
                          # on each iter.
train=true            # use --train false to run only diagnostics.




# End configuration section



echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;


if [ $# != 2 ]; then
  echo "Usage: $0  [options] <egs-dir>  <model-dir>"
  echo " e.g.: $0 exp/chaina/tdnn1a_sp/egs  exp/chaina/tdnn1a_sp"
  echo ""
  echo " TODO: more documentation"
  exit 1
fi

egs_dir=$1
dir=$2

set -e -u  # die on failed command or undefined variable

steps/chaina/validate_randomized_egs.sh $egs_dir

for f in $dir/init/info.txt $dir/init/bottom.raw; do
  if [ ! -f $f ]; then
    echo "$0: expected file $f to exist"
    exit 1
  fi
done


frame_subsampling_factor=$(awk '/^frame_subsampling_factor/ {print $2}' <$dir/init/info.txt)
num_scp_files=$(awk '/^num_scp_files/ {print $2}' <$dir/egs/info.txt)

steps/chaina/internal/get_train_schedule.py \
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



if [ "$use_gpu" != "no" ]; then gpu_cmd_opt="--gpu 1"; else gpu_cmd_opt=""; fi

num_iters=$(wc -l <$dir/schedule.txt)

echo "$0: will train for $num_epochs epochs = $num_iters iterations"

# source the 1st line of schedule.txt in the shell; this sets
# lrate and dropout_opt, among other variables.
. <(head -n 1 $dir/schedule.txt)
langs=$(awk '/^langs/ { $1=""; print; }' <$dir/init/info.txt)

mkdir -p $dir/log

# Copy models with initial learning rate and dropout options from $dir/init to $dir/0
mkdir -p $dir/0
run.pl $dir/log/init_bottom_model.log \
  nnet3-copy --learning-rate=$lrate $dropout_opt $dir/init/bottom.raw $dir/0/bottom.raw
for lang in $langs; do
  run.pl $dir/log/init_model_$lang.log \
      nnet3-am-copy --learning-rate=$lrate $dropout_opt $dir/init/$lang.mdl $dir/0/$lang.mdl
done


x=0
if [ $stage -gt $x ]; then x=$stage; fi

while [ $x -lt $num_iters ]; do
  # Source some variables fromm schedule.txt.  The effect will be something
  # like the following:
  # iter=0; num_jobs=2; inv_num_jobs=0.5; scp_indexes=(pad 1 2); frame_shifts=(pad 1 2); dropout_opt="--edits='set-dropout-proportion name=* proportion=0.0'" lrate=0.002
  . <(grep "^iter=$x;" $dir/schedule.txt)

  echo "$0: training, iteration $x, num-jobs is $num_jobs"

  next_x=$[$x+1]
  model_in_dir=$dir/$x
  if [ ! -f $model_in_dir/bottom.raw ]; then
    echo "$0: expected $model_in_dir/bottom.raw to exist"
    exit 1
  fi
  den_fst_dir=$egs_dir/misc
  transform_dir=$dir/init
  model_out_dir=$dir/${next_x}


  # for the first 4 iterations, plus every $diagnostic_period iterations, launch
  # some diagnostic processes.  We don't do this on iteration 0, because
  # the batchnorm stats wouldn't be ready
  if [ $x -gt 0 ] && [ $[x%diagnostic_period] -eq 0 -o $x -lt 5 ]; then

    [ -f $dir/$x/.error_diagnostic ] && rm $dir/$x/.error_diagnostic
    for name in train heldout; do
      $cmd $gpu_cmd_opt $dir/log/diagnostic_${name}.$x.log \
         nnet3-chaina-train --use-gpu=$use_gpu \
            --bottom-model-test-mode=true --top-model-test-mode=true
            --leaky-hmm-coefficient=$leaky_hmm_coefficient \
            --xent-regularize=$xent_regularize \
            --print-interval=10  \
           $model_in_dir $den_fst_dir $transform_dir \
           "ark:nnet3-chain-merge-egs --minibatch-size=$groups_per_minibatch scp:$egs_dir/${name}_subset.scp ark:-|" \
      || touch $dir/$x/.error_diagnostic &
    done
  fi

  if $train; then
    if [ -d $dir/$next_x ]; then
      echo "$0: removing previous contents of $dir/$next_x"
      rm -r $dir/$next_x
    fi
    mkdir -p $dir/$next_x

    for j in $(seq $num_jobs); do
      scp_index=${scp_indexes[$j]}
      frame_shift=${frame_shifts[$j]}

      $cmd $gpu_cmd_opt $dir/log/train.$x.$j.log \
           nnet3-chaina-train --job-id=$j --use-gpu=$use_gpu --apply-deriv-weights=$apply_deriv_weights \
           --leaky-hmm-coefficient=$leaky_hmm_coefficient --xent-regularize=$xent_regularize \
           --print-interval=10 --max-param-change=$max_param_change \
           --l2-regularize-factor=$inv_num_jobs --optimization.memory-compression-level=$memory_compression_level \
           $model_in_dir $den_fst_dir $transform_dir \
           "ark:nnet3-chain-copy-egs --frame-shift=$frame_shift scp:$egs_dir/train.$scp_index.scp ark:- | nnet3-chain-shuffle-egs --buffer-size=$shuffle_buffer_size --srand=$x ark:- ark:- | nnet3-chain-merge-egs --minibatch-size=$groups_per_minibatch ark:- ark:-|" \
           $model_out_dir || touch $dir/$next_x/.error &
    done
    wait
    if [ -f $dir/$next_x/.error ]; then
      echo "$0: error detected training on iteration $x"
      exit 1
    fi
    # First average the bottom models
    models=$(for j in $(seq $num_jobs); do echo $dir/$next_x/bottom.$j.raw; done)
    run.pl $dir/log/average.$x.log \
           nnet3-average $models - \| \
           nnet3-copy --learning-rate=$lrate $dropout_opt - $dir/$next_x/bottom.raw
    rm $models
    for lang in $langs; do
      models=$dir/$next_x/$lang.*.raw
      run.pl $dir/log/average_${lang}.$x.log \
             nnet3-average $models - \| \
             nnet3-am-copy --set-raw-nnet=- --learning-rate=$lrate $dropout_opt $dir/$iter/$lang.mdl $dir/$next_x/$lang.mdl
      rm $models
    done
  fi

  wait
  if [ -f $dir/$x/.error_diagnostic ]; then
    echo "$0: error detected in diagnostics on iteration $x"
    exit 1
  fi

  # TODO: diagnostics; cleanup
  x=$[x+1]
done

# TODO: later we'll have a model combination phase.

if [ $stage -le $num_iters ] && $train; then
  # Now accumulate the class-dependent mean (and variance) stats of the
  # adaptation model, which will be needed for decoding.  We remove the map that
  # had reduced the num-classes from several thousand to (e.g.) 200, because we
  # are now estimating the means on a larger set of data and we're not concerned
  # about noisy estimates.
  mkdir -p $dir/transforms_unmapped
  # Note: the plan was to add the option --remove-pdf-map=true to the 'copy'
  # command below (to use the full number of pdf-ids as classes in test time),
  # but it seemed to degrade the objective function, based on diagnostics.
  # We'll look into this later.
  for lang in $langs; do
    run.pl $dir/log/copy_transform_${lang}.log \
        nnet3-adapt copy $dir/init/${lang}.ada $dir/transforms_unmapped/${lang}.ada
  done
  if [ -d $dir/final ]; then
    echo "$0: removing previous contents of $dir/final"
    rm -r $dir/final
  fi
  mkdir -p $dir/final
  den_fst_dir=$egs_dir/misc

  $cmd $gpu_cmd_opt JOB=1:$num_scp_files $dir/log/acc_target_model.JOB.log \
    nnet3-chaina-train --job-id=JOB --use-gpu=$use_gpu \
      --print-interval=10 \
      --bottom-model-test-mode=true --top-model-test-mode=true \
      --adaptation-model-accumulate=true \
         $dir/$num_iters $den_fst_dir $dir/transforms_unmapped \
        "ark:nnet3-chain-shuffle-egs --buffer-size=$shuffle_buffer_size scp:$egs_dir/train.JOB.scp ark:- | nnet3-chain-merge-egs --minibatch-size=$groups_per_minibatch ark:- ark:-|" \
        $dir/final

  for lang in $langs; do
    stats=$dir/final/${lang}.*.ada
    run.pl $dir/log/estimate_target_model_${lang}.log \
           nnet3-adapt estimate $stats $dir/final/${lang}.ada
    #rm $stats
  done
  cp $dir/$num_iters/bottom.raw $dir/$num_iters/*.mdl $dir/final
fi

if [ $stage -le $[num_iters+1] ]; then
  # Accumulate some final diagnostics.  The difference with the last iteration's
  # diagnostics is that we use test-mode for the adaptation model (i.e. a target
  # model computed from all the data, not just one minibatch).
  [ -f $dir/final/.error_diagnostic ] && rm $dir/final/.error_diagnostic
  for name in train heldout; do
    den_fst_dir=$egs_dir/misc
    $cmd $gpu_cmd_opt $dir/log/diagnostic_${name}.final.log \
       nnet3-chaina-train --use-gpu=$use_gpu \
         --bottom-model-test-mode=true --top-model-test-mode=true \
         --adaptation-test-mode=true \
         --leaky-hmm-coefficient=$leaky_hmm_coefficient \
         --xent-regularize=$xent_regularize \
         --print-interval=10  \
          $dir/final $den_fst_dir $dir/final \
           "ark:nnet3-chain-merge-egs --minibatch-size=$groups_per_minibatch scp:$egs_dir/${name}_subset.scp ark:-|" \
      || touch $dir/final/.error_diagnostic &
  done
  wait
  if [ -f $dir/final/.error_diagnostic ]; then
    echo "$0: error getting final diagnostic information"
    exit 1
  fi
fi


transform_dir=$dir/init

echo "$0: done"
exit 0
