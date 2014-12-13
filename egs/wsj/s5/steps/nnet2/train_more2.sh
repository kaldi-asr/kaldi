#!/bin/bash

# Copyright 2014  Johns Hopkins University (Author: Daniel Povey). 
# Apache 2.0.

# This script further trains an already-existing neural network,
# given an existing model and an examples (egs/) directory.
# This version of the script epects an egs/ directory in the newer
# format, as created by get_egs2.sh.
#

# Begin configuration section.
cmd=run.pl
num_epochs=10      # Number of epochs of training; number of iterations is
                   # worked out from this.
num_iters_final=20 # Maximum number of final iterations to give to the
                  # optimization over the validation set.
learning_rate_factor=1.0 # You can use this to gradually decrease the learning
                         # rate during training (e.g. use 0.2); the initial
                         # learning rates are as specified in the model, but it
                         # will decrease slightly on each iteration to achieve
                         # this ratio.

combine=true # controls whether or not to do the final model combination.
combine_regularizer=1.0e-14 # Small regularizer so that parameters won't go crazy.
max_models_combine=20 # The "max_models_combine" is the maximum number of models we give
  # to the final 'combine' stage, but these models will themselves be averages of
  # iteration-number ranges.

minibatch_size=128 # by default use a smallish minibatch size for neural net
                   # training; this controls instability which would otherwise
                   # be a problem with multi-threaded update.  Note: it also
                   # interacts with the "preconditioned" update which generally
                   # works better with larger minibatch size, so it's not
                   # completely cost free.

shuffle_buffer_size=5000 # This "buffer_size" variable controls randomization of the samples
                # on each iter.  You could set it to 0 or to a large value for complete
                # randomization, but this would both consume memory and cause spikes in
                # disk I/O.  Smaller is easier on disk and memory but less random.  It's
                # not a huge deal though, as samples are anyway randomized right at the start.
num_jobs_nnet=4
mix_up=0
stage=-5
num_threads=16
parallel_opts="-pe smp 16 -l ram_free=1G,mem_free=1G" # by default we use 16 threads; this lets the queue know.
   # note: parallel_opts doesn't automatically get adjusted if you adjust num-threads.
combine_num_threads=8
cleanup=true
prior_subset_size=10000 # 10k samples per job, for computing priors.  Should be
                        # more than enough.
num_jobs_compute_prior=10 # these are single-threaded, run on CPU.
remove_egs=false
# End configuration section.


echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 3 ]; then
  echo "Usage: $0 [opts] <input-model> <egs-dir> <exp-dir>"
  echo " e.g.: $0 exp/nnet4c/final.mdl exp/nnet4c/egs exp/nnet5c/"
  echo "see also the older script update_nnet.sh which creates the egs itself"
  echo ""
  echo "Main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config file containing options"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --num-epochs <#epochs|15>                        # Number of epochs of training"
  echo "                                                   # while reducing learning rate (determines #iterations, together"
  echo "                                                   # with --samples-per-iter and --num-jobs-nnet)"
  echo "  --num-jobs-nnet <#jobs|4>                        # Number of neural-net jobs to run in parallel"
  echo "  --learning-rate-factor<factor|1.0>               # Factor (e.g. 0.2) by which to change learning rate"
  echo "                                                   # during the course of training"
  echo "  --num-threads <num-threads|16>                   # Number of parallel threads per job (will affect results"
  echo "                                                   # as well as speed; may interact with batch size; if you increase"
  echo "                                                   # this, you may want to decrease the batch size."
  echo "  --parallel-opts <opts|\"-pe smp 16 -l ram_free=1G,mem_free=1G\">      # extra options to pass to e.g. queue.pl for processes that"
  echo "                                                   # use multiple threads... note, you might have to reduce mem_free,ram_free"
  echo "                                                   # versus your defaults, because it gets multiplied by the -pe smp argument."
  echo "  --minibatch-size <minibatch-size|128>            # Size of minibatch to process (note: product with --num-threads"
  echo "                                                   # should not get too large, e.g. >2k)."
  echo "  --num-iters-final <#iters|20>                    # Number of final iterations to give to nnet-combine-fast to "
  echo "                                                   # interpolate parameters (the weights are learned with a validation set)"
  echo "  --mix-up <#mix|0>                                # If specified, add quasi-targets, analogous to a mixture of Gaussians vs."
  echo "                                                   # single Gaussians.  Only do this if not already mixed-up."
  echo "  --combine <true or false|true>                   # If true, do the final nnet-combine-fast stage."
  echo "  --stage <stage|-5>                               # Used to run a partially-completed training process from somewhere in"
  echo "                                                   # the middle."  
  exit 1;
fi

input_mdl=$1
egs_dir=$2
dir=$3

# Check some files.
for f in $input_mdl $egs_dir/egs.1.ark; do
  [ ! -f $f ] && echo "$0: expected file $f to exist." && exit 1;
done

mkdir -p $dir/log

# Copy some things from the directory where the input model is located, to the
# experimental directory, if they exist.  These might be needed for things like
# decoding.
input_dir=$(dirname $input_mdl);
for f in tree splice_opts cmvn_opts final.mat; do
  if [ -f $input_dir/$f ]; then
    cp $input_dir/$f $dir/
  fi
done

frames_per_eg=$(cat $egs_dir/info/frames_per_eg) || { echo "error: no such file $egs_dir/info/frames_per_eg"; exit 1; }
num_archives=$(cat $egs_dir/info/num_archives) || { echo "error: no such file $egs_dir/info/frames_per_eg"; exit 1; }

# num_archives_expanded considers each separate label-position from
# 0..frames_per_eg-1 to be a separate archive.
num_archives_expanded=$[$num_archives*$frames_per_eg]

if [ $num_jobs_nnet -gt $num_archives_expanded ]; then
  echo "$0: --num-jobs-nnet cannot exceed num-archives*frames-per-eg which is $num_archives_expanded"
  echo "$0: setting --num-jobs-nnet to $num_archives_expanded"
  num_jobs_nnet=$num_archives_expanded
fi


# set num_iters so that as close as possible, we process the data $num_epochs
# times, i.e. $num_iters*$num_jobs_nnet == $num_epochs*$num_archives_expanded
num_iters=$[($num_epochs*$num_archives_expanded)/$num_jobs_nnet]

echo "$0: Will train for $num_epochs epochs = $num_iters iterations"

per_iter_learning_rate_factor=$(perl -e "print ($learning_rate_factor ** (1.0 / $num_iters));")

mix_up_iter=$[$num_iters/4]  # mix up after only a short way into training, as
                             # most likely the net is already quite well trained.

if [ $num_threads -eq 1 ]; then
  parallel_suffix="-simple" # this enables us to use GPU code if
                         # we have just one thread.
  parallel_train_opts=
  if ! cuda-compiled; then
    echo "$0: WARNING: you are running with one thread but you have not compiled"
    echo "   for CUDA.  You may be running a setup optimized for GPUs.  If you have"
    echo "   GPUs and have nvcc installed, go to src/ and do ./configure; make"
  fi
else
  parallel_suffix="-parallel"
  parallel_train_opts="--num-threads=$num_threads"
fi


approx_iters_per_epoch=$[$num_iters/$num_epochs]
# First work out how many models we want to combine over in the final
# nnet-combine-fast invocation.  This equals
# min(max(max_models_combine, iters_per_epoch),
#     2/3 * iters_after_mixup)
num_models_combine=$max_models_combine
if [ $num_models_combine -lt $approx_iters_per_epoch ]; then
  num_models_combine=$approx_iters_per_epoch
fi
iters_after_mixup_23=$[(($num_iters-$mix_up_iter-1)*2)/3]
if [ $num_models_combine -gt $iters_after_mixup_23 ]; then
  num_models_combine=$iters_after_mixup_23
fi
first_model_combine=$[$num_iters-$num_models_combine+1]

cp $input_mdl $dir/0.mdl || exit 1;

x=0

while [ $x -lt $num_iters ]; do
  if [ $x -ge 0 ] && [ $stage -le $x ]; then
    # Set off jobs doing some diagnostics, in the background.
    $cmd $dir/log/compute_prob_valid.$x.log \
      nnet-compute-prob $dir/$x.mdl ark:$egs_dir/valid_diagnostic.egs &
    $cmd $dir/log/compute_prob_train.$x.log \
      nnet-compute-prob $dir/$x.mdl ark:$egs_dir/train_diagnostic.egs &
    if [ $x -gt 0 ] && [ ! -f $dir/log/mix_up.$[$x-1].log ]; then
      $cmd $dir/log/progress.$x.log \
        nnet-show-progress --use-gpu=no $dir/$[$x-1].mdl $dir/$x.mdl ark:$egs_dir/train_diagnostic.egs &
    fi
    
    echo "Training neural net (pass $x)"

    rm $dir/.error 2>/dev/null
    ( # this sub-shell is so that when we "wait" below,
      # we only wait for the training jobs that we just spawned,
      # not the diagnostic jobs that we spawned above.
      
      # We can't easily use a single parallel SGE job to do the main training,
      # because the computation of which archive and which --frame option
      # to use for each job is a little complex, so we spawn each one separately.
      for n in $(seq $num_jobs_nnet); do
        k=$[$x*$num_jobs_nnet + $n - 1]; # k is a zero-based index that we'll derive
                                         # the other indexes from.
        archive=$[($k%$num_archives)+1]; # work out the 1-based archive index.
        frame=$[(($k/$num_archives)%$frames_per_eg)]; # work out the 0-based frame
        # index; this increases more slowly than the archive index because the
        # same archive with different frame indexes will give similar gradients,
        # so we want to separate them in time.

        $cmd $parallel_opts $dir/log/train.$x.$n.log \
          nnet-train$parallel_suffix $parallel_train_opts \
          --minibatch-size=$minibatch_size --srand=$x $dir/$x.mdl \
          "ark:nnet-copy-egs --frame=$frame ark:$egs_dir/egs.$archive.ark ark:-|nnet-shuffle-egs --buffer-size=$shuffle_buffer_size --srand=$x ark:- ark:-|" \
          $dir/$[$x+1].$n.mdl || touch $dir/.error &
      done
      wait
    )
    # the error message below is not that informative, but $cmd will
    # have printed a more specific one.
    [ -f $dir/.error ] && echo "$0: error on iteration $x of training" && exit 1;

    nnets_list=
    for n in `seq 1 $num_jobs_nnet`; do
      nnets_list="$nnets_list $dir/$[$x+1].$n.mdl"
    done     

    $cmd $dir/log/average.$x.log \
      nnet-am-average $nnets_list - \| \
      nnet-am-copy --learning-rate-factor=$per_iter_learning_rate_factor - $dir/$[$x+1].mdl || exit 1;

    if [ "$mix_up" -gt 0 ] && [ $x -eq $mix_up_iter ]; then
      # mix up.
      echo Mixing up from $num_leaves to $mix_up components
      $cmd $dir/log/mix_up.$x.log \
        nnet-am-mixup --min-count=10 --num-mixtures=$mix_up \
         $dir/$[$x+1].mdl $dir/$[$x+1].mdl || exit 1;
    fi
    rm $nnets_list
  fi
  x=$[$x+1]
done


if [ $stage -le $num_iters ]; then
  echo "Doing final combination to produce final.mdl"

  # Now do combination.
  nnets_list=()
  # the if..else..fi statement below sets 'nnets_list'.
  if [ $max_models_combine -lt $num_models_combine ]; then
    # The number of models to combine is too large, e.g. > 20.  In this case,
    # each argument to nnet-combine-fast will be an average of multiple models.
    cur_offset=0 # current offset from first_model_combine.
    for n in $(seq $max_models_combine); do
      next_offset=$[($n*$num_models_combine)/$max_models_combine]
      sub_list="" 
      for o in $(seq $cur_offset $[$next_offset-1]); do
        iter=$[$first_model_combine+$o]
        mdl=$dir/$iter.mdl
        [ ! -f $mdl ] && echo "Expected $mdl to exist" && exit 1;
        sub_list="$sub_list $mdl"
      done
      nnets_list[$[$n-1]]="nnet-am-average $sub_list - |"
      cur_offset=$next_offset
    done
  else
    nnets_list=
    for n in $(seq 0 $[num_models_combine-1]); do
      iter=$[$first_model_combine+$n]
      mdl=$dir/$iter.mdl
      [ ! -f $mdl ] && echo "Expected $mdl to exist" && exit 1;
      nnets_list[$n]=$mdl
    done
  fi


  # Below, use --use-gpu=no to disable nnet-combine-fast from using a GPU, as
  # if there are many models it can give out-of-memory error; set num-threads to 8
  # to speed it up (this isn't ideal...)
  num_egs=`nnet-copy-egs ark:$egs_dir/combine.egs ark:/dev/null 2>&1 | tail -n 1 | awk '{print $NF}'`
  mb=$[($num_egs+$combine_num_threads-1)/$combine_num_threads]
  [ $mb -gt 512 ] && mb=512
  # Setting --initial-model to a large value makes it initialize the combination
  # with the average of all the models.  It's important not to start with a
  # single model, or, due to the invariance to scaling that these nonlinearities
  # give us, we get zero diagonal entries in the fisher matrix that
  # nnet-combine-fast uses for scaling, which after flooring and inversion, has
  # the effect that the initial model chosen gets much higher learning rates
  # than the others.  This prevents the optimization from working well.
  $cmd $combine_parallel_opts $dir/log/combine.log \
    nnet-combine-fast --initial-model=100000 --num-lbfgs-iters=40 --use-gpu=no \
      --num-threads=$combine_num_threads \
      --verbose=3 --minibatch-size=$mb "${nnets_list[@]}" ark:$egs_dir/combine.egs \
      $dir/final.mdl || exit 1;

  # Normalize stddev for affine or block affine layers that are followed by a
  # pnorm layer and then a normalize layer.
  $cmd $dir/log/normalize.log \
    nnet-normalize-stddev $dir/final.mdl $dir/final.mdl || exit 1;

  # Compute the probability of the final, combined model with
  # the same subset we used for the previous compute_probs, as the
  # different subsets will lead to different probs.
  $cmd $dir/log/compute_prob_valid.final.log \
    nnet-compute-prob $dir/final.mdl ark:$egs_dir/valid_diagnostic.egs &
  $cmd $dir/log/compute_prob_train.final.log \
    nnet-compute-prob $dir/final.mdl ark:$egs_dir/train_diagnostic.egs &
fi

if [ $stage -le $[$num_iters+1] ]; then
  echo "Getting average posterior for purposes of adjusting the priors."
  # Note: this just uses CPUs, using a smallish subset of data.
  rm $dir/post.$x.*.vec 2>/dev/null
  $cmd JOB=1:$num_jobs_compute_prior $dir/log/get_post.$x.JOB.log \
    nnet-copy-egs --frame=random --srand=JOB ark:$egs_dir/egs.1.ark ark:- \| \
    nnet-subset-egs --srand=JOB --n=$prior_subset_size ark:- ark:- \| \
    nnet-compute-from-egs "nnet-to-raw-nnet $dir/final.mdl -|" ark:- ark:- \| \
    matrix-sum-rows ark:- ark:- \| vector-sum ark:- $dir/post.$x.JOB.vec || exit 1;

  sleep 3;  # make sure there is time for $dir/post.$x.*.vec to appear.

  $cmd $dir/log/vector_sum.$x.log \
   vector-sum $dir/post.$x.*.vec $dir/post.$x.vec || exit 1;

  rm $dir/post.$x.*.vec;

  echo "Re-adjusting priors based on computed posteriors"
  $cmd $dir/log/adjust_priors.final.log \
    nnet-adjust-priors $dir/final.mdl $dir/post.$x.vec $dir/final.mdl || exit 1;
fi


if [ ! -f $dir/final.mdl ]; then
  echo "$0: $dir/final.mdl does not exist."
  # we don't want to clean up if the training didn't succeed.
  exit 1;
fi

sleep 2

echo Done

if $cleanup; then
  echo Cleaning up data
  if $remove_egs && [[ $egs_dir =~ $dir/egs* ]]; then
    steps/nnet2/remove_egs.sh $egs_dir
  fi

  echo Removing most of the models
  for x in `seq 0 $num_iters`; do
    if [ $[$x%100] -ne 0 ] && [ $x -ne $num_iters ] && [ -f $dir/$x.mdl ]; then
       # delete all but every 100th model; don't delete the ones which combine to form the final model.
      rm $dir/$x.mdl
    fi
  done
fi

