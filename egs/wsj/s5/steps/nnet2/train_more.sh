#!/bin/bash

# Copyright 2014  Johns Hopkins University (Author: Daniel Povey). 
# Apache 2.0.


# This script further trains an already-existing neural network,
# given an existing model and an examples (egs/) directory.
# The number of parallel jobs (--num-jobs-nnet) is determined by the
# egs directory.

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
mix_up=0
stage=-5
num_threads=16
parallel_opts="--num-threads 16 --mem 1G" # by default we use 16 threads; this lets the queue know.
   # note: parallel_opts doesn't automatically get adjusted if you adjust num-threads.
cleanup=true
remove_egs=false
prior_subset_size=10000 # 10k samples per job, for computing priors.  Should be
                        # more than enough.
# End configuration section.


echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 3 ]; then
  echo "Usage: $0 [opts] <input-model> <egs-dir> <exp-dir>"
  echo " e.g.: $0 exp/nnet4c/final.mdl exp/nnet4c/egs exp/nnet5c/"
  echo "see also the older script update_nnet.sh which creates the egs itself"
  echo "You probably now want to use train_more2.sh, which uses the newer,"
  echo "more compact egs format."
  echo ""
  echo "Main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config file containing options"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --num-epochs <#epochs|15>                        # Number of epochs of training"
  echo "                                                   # while reducing learning rate (determines #iterations, together"
  echo "                                                   # with --samples-per-iter and --num-jobs-nnet)"
  echo "  --learning-rate-factor<factor|1.0>               # Factor (e.g. 0.2) by which to change learning rate"
  echo "                                                   # during the course of training"
  echo "  --num-threads <num-threads|16>                   # Number of parallel threads per job (will affect results"
  echo "                                                   # as well as speed; may interact with batch size; if you increase"
  echo "                                                   # this, you may want to decrease the batch size."
  echo "  --parallel-opts <opts|\"--num-threads 16 --mem 1G\">      # extra options to pass to e.g. queue.pl for processes that"
  echo "                                                   # use multiple threads... "
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
for f in $input_mdl $egs_dir/egs.1.0.ark; do
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

iters_per_epoch=$(cat $egs_dir/iters_per_epoch) || exit 1;
num_jobs_nnet=$(cat $egs_dir/num_jobs_nnet) || exit 1;

num_iters=$[$num_epochs * $iters_per_epoch];
per_iter_learning_rate_factor=$(perl -e "print ($learning_rate_factor ** (1.0 / $num_iters));")

echo "$0: Will train for $num_epochs epochs, equalling $num_iters iterations."

mix_up_iter=$[$num_iters/2]

if [ $num_threads -eq 1 ]; then
  train_suffix="-simple" # this enables us to use GPU code if
                         # we have just one thread.
  if ! cuda-compiled; then
    echo "$0: WARNING: you are running with one thread but you have not compiled"
    echo "   for CUDA.  You may be running a setup optimized for GPUs.  If you have"
    echo "   GPUs and have nvcc installed, go to src/ and do ./configure; make"
  fi
else
  train_suffix="-parallel --num-threads=$num_threads"
fi

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


    $cmd $parallel_opts JOB=1:$num_jobs_nnet $dir/log/train.$x.JOB.log \
      nnet-shuffle-egs --buffer-size=$shuffle_buffer_size --srand=$x \
      ark:$egs_dir/egs.JOB.$[$x%$iters_per_epoch].ark ark:- \| \
      nnet-train$train_suffix --minibatch-size=$minibatch_size --srand=$x $dir/$x.mdl \
        ark:- $dir/$[$x+1].JOB.mdl \
      || exit 1;

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

# Now do combination.
# At the end, final.mdl will be a combination of the last e.g. 10 models.
nnets_list=()
[ $num_iters_final -gt $num_iters ] && num_iters_final=$num_iters
[ "$mix_up" -gt 0 ] && [ $num_iters_final -gt $[$num_iters-$mix_up_iter] ] && \
  num_iters_final=$[$num_iters-$mix_up_iter]

start=$[$num_iters-$num_iters_final+1]
for x in `seq $start $num_iters`; do
  idx=$[$x-$start]
  if [ $x -gt $mix_up_iter ]; then
    nnets_list[$idx]=$dir/$x.mdl # "nnet-am-copy --remove-dropout=true $dir/$x.mdl - |"
  fi
done

if [ $stage -le $num_iters ]; then
  if $combine; then
    echo "Doing final combination to produce final.mdl"
  # Below, use --use-gpu=no to disable nnet-combine-fast from using a GPU, as
  # if there are many models it can give out-of-memory error; set num-threads to 8
  # to speed it up (this isn't ideal...)
    this_num_threads=$num_threads
    [ $this_num_threads -lt 8 ] && this_num_threads=8
    num_egs=`nnet-copy-egs ark:$egs_dir/combine.egs ark:/dev/null 2>&1 | tail -n 1 | awk '{print $NF}'`
    mb=$[($num_egs+$this_num_threads-1)/$this_num_threads]
    [ $mb -gt 512 ] && mb=512
  # Setting --initial-model to a large value makes it initialize the combination
  # with the average of all the models.  It's important not to start with a
  # single model, or, due to the invariance to scaling that these nonlinearities
  # give us, we get zero diagonal entries in the fisher matrix that
  # nnet-combine-fast uses for scaling, which after flooring and inversion, has
  # the effect that the initial model chosen gets much higher learning rates
  # than the others.  This prevents the optimization from working well.
    $cmd $parallel_opts $dir/log/combine.log \
      nnet-combine-fast --initial-model=100000 --num-lbfgs-iters=40 --use-gpu=no \
      --num-threads=$this_num_threads --regularizer=$combine_regularizer \
      --verbose=3 --minibatch-size=$mb "${nnets_list[@]}" ark:$egs_dir/combine.egs \
      $dir/final.mdl || exit 1;

  # Normalize stddev for affine or block affine layers that are followed by a
  # pnorm layer and then a normalize layer.
    $cmd $parallel_opts $dir/log/normalize.log \
      nnet-normalize-stddev $dir/final.mdl $dir/final.mdl || exit 1;

  # Compute the probability of the final, combined model with
  # the same subset we used for the previous compute_probs, as the
  # different subsets will lead to different probs.
    $cmd $dir/log/compute_prob_valid.final.log \
      nnet-compute-prob $dir/final.mdl ark:$egs_dir/valid_diagnostic.egs &
    $cmd $dir/log/compute_prob_train.final.log \
      nnet-compute-prob $dir/final.mdl ark:$egs_dir/train_diagnostic.egs &
  else
    echo "$0: --combine=false so just using last model."
    cp $dir/$x.mdl $dir/final.mdl
  fi
fi

if [ $stage -le $[$num_iters+1] ]; then
  echo "Getting average posterior for purposes of adjusting the priors."
  # Note: this just uses CPUs, using a smallish subset of data.
  rm $dir/post.*.vec 2>/dev/null
  $cmd JOB=1:$num_jobs_nnet $dir/log/get_post.JOB.log \
    nnet-subset-egs --n=$prior_subset_size ark:$egs_dir/egs.JOB.0.ark ark:- \| \
    nnet-compute-from-egs "nnet-to-raw-nnet $dir/final.mdl -|" ark:- ark:- \| \
    matrix-sum-rows ark:- ark:- \| vector-sum ark:- $dir/post.JOB.vec || exit 1;

  sleep 3;  # make sure there is time for $dir/post.*.vec to appear.

  $cmd $dir/log/vector_sum.log \
   vector-sum $dir/post.*.vec $dir/post.vec || exit 1;

  rm $dir/post.*.vec;

  echo "Re-adjusting priors based on computed posteriors"
  $cmd $dir/log/adjust_priors.log \
    nnet-adjust-priors $dir/final.mdl $dir/post.vec $dir/final.mdl || exit 1;
fi


sleep 2

echo Done


$remove_egs && steps/nnet2/remove_egs.sh $dir/egs

if $cleanup; then
  echo Removing most of the models
  for x in `seq 0 $num_iters`; do
    if [ $[$x%100] -ne 0 ] && [ $x -lt $[$num_iters-$num_iters_final+1] ]; then 
       # delete all but every 100th model; don't delete the ones which combine to form the final model.
      rm $dir/$x.mdl
    fi
  done
fi
