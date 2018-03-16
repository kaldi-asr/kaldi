#!/bin/bash

# Copyright 2012  Johns Hopkins University (Author: Daniel Povey).  Apache 2.0.

# This script is for training networks with tanh nonlinearities; it starts with
# a given model and supports increasing the hidden-layer dimension.  It is
# otherwise similar to train_tanh.sh

# Begin configuration section.
cmd=run.pl
num_epochs=15      # Number of epochs during which we reduce
                   # the learning rate; number of iteration is worked out from this.
num_epochs_extra=5 # Number of epochs after we stop reducing
                   # the learning rate.
num_iters_final=20 # Maximum number of final iterations to give to the
                   # optimization over the validation set.
initial_learning_rate=0.04
final_learning_rate=0.004
softmax_learning_rate_factor=0.5 # Train this layer half as fast as the other layers.


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


stage=-5


mix_up=0 # Number of components to mix up to (should be > #tree leaves, if
         # specified.)  Will do this at the start.
widen=0 # If specified, it will increase the hidden-layer dimension 
                            # to this value.  Will do this at the start.
bias_stddev=0.5 # will be used for widen

num_threads=16
parallel_opts="--num-threads $num_threads"  # using a smallish #threads by default, out of stability concerns.
  # note: parallel_opts doesn't automatically get adjusted if you adjust num-threads.
cleanup=true
# End configuration section.


echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 3 ]; then
  echo "Usage: $0 [opts] <egs-dir> <old-nnet-dir> <exp-dir>"
  echo " e.g.: $0 --widen 1024 exp/tri4_nnet/egs exp/tri4_nnet exp/tri5_nnet"
  echo ""
  echo "Main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config file containing options"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --num-epochs <#epochs|15>                        # Number of epochs of main training"
  echo "                                                   # while reducing learning rate (determines #iterations, together"
  echo "                                                   # with --samples-per-iter and --num-jobs-nnet)"
  echo "  --num-epochs-extra <#epochs-extra|5>             # Number of extra epochs of training"
  echo "                                                   # after learning rate fully reduced"
  echo "  --initial-learning-rate <initial-learning-rate|0.02> # Learning rate at start of training, e.g. 0.02 for small"
  echo "                                                       # data, 0.01 for large data"
  echo "  --final-learning-rate  <final-learning-rate|0.004>   # Learning rate at end of training, e.g. 0.004 for small"
  echo "                                                   # data, 0.001 for large data"
  echo "  --mix-up <#pseudo-gaussians|0>                   # Can be used to have multiple targets in final output layer,"
  echo "                                                   # per context-dependent state.  Try a number several times #states."
  echo "  --num-threads <num-threads|16>                   # Number of parallel threads per job (will affect results"
  echo "                                                   # as well as speed; may interact with batch size; if you increase"
  echo "                                                   # this, you may want to decrease the batch size."
  echo "  --parallel-opts <opts|\"--num-threads 16\">            # extra options to pass to e.g. queue.pl for processes that"
  echo "                                                   # use multiple threads."
  echo "  --minibatch-size <minibatch-size|128>            # Size of minibatch to process (note: product with --num-threads"
  echo "                                                   # should not get too large, e.g. >2k)."
  echo "  --num-iters-final <#iters|10>                    # Number of final iterations to give to nnet-combine-fast to "
  echo "                                                   # interpolate parameters (the weights are learned with a validation set)"
  echo "  --stage <stage|-5>                               # Used to run a partially-completed training process from somewhere in"
  echo "                                                   # the middle."
  exit 1;
fi

egs_dir=$1
nnet_dir=$2
dir=$3

# Check some files.
for f in $egs_dir/egs.1.0.ark $nnet_dir/final.mdl; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

num_jobs_nnet=`cat $egs_dir/num_jobs_nnet` || exit 1;
iters_per_epoch=`cat $egs_dir/iters_per_epoch`  || exit 1;

mkdir -p $dir/log

cp $nnet_dir/phones.txt $dir 2>/dev/null

cp $nnet_dir/splice_opts $dir 2>/dev/null
cp $nnet_dir/final.mat $dir 2>/dev/null # any LDA matrix...
cp $nnet_dir/tree $dir


if [ $stage -le -2 ] && [ $mix_up -gt 0 ]; then
  echo Mixing up to $mix_up components
  $cmd $dir/log/mix_up.$x.log \
    nnet-am-mixup --min-count=10 --num-mixtures=$mix_up \
      $nnet_dir/final.mdl $dir/0.mdl || exit 1;
else 
  cp $nnet_dir/final.mdl $dir/0.mdl || exit 1;
fi

if [ $stage -le -1 ] && [ $widen -gt 0 ]; then
  echo "$0: Widening nnet to hidden-layer-dim=$widen"
  $cmd $dir/log/widen.log \
    nnet-am-widen --hidden-layer-dim=$widen $dir/0.mdl $dir/0.mdl || exit 1;
fi

num_iters_reduce=$[$num_epochs * $iters_per_epoch];
num_iters_extra=$[$num_epochs_extra * $iters_per_epoch];
num_iters=$[$num_iters_reduce+$num_iters_extra]

echo "$0: Will train for $num_epochs + $num_epochs_extra epochs, equalling "
echo "$0: $num_iters_reduce + $num_iters_extra = $num_iters iterations, "
echo "$0: (while reducing learning rate) + (with constant learning rate)."

x=0
while [ $x -lt $num_iters ]; do
  if [ $x -ge 0 ] && [ $stage -le $x ]; then
    # Set off jobs doing some diagnostics, in the background.
    $cmd $dir/log/compute_prob_valid.$x.log \
      nnet-compute-prob $dir/$x.mdl ark:$egs_dir/valid_diagnostic.egs &
    $cmd $dir/log/compute_prob_train.$x.log \
      nnet-compute-prob $dir/$x.mdl ark:$egs_dir/train_diagnostic.egs &

    echo "Training neural net (pass $x)"

    $cmd $parallel_opts JOB=1:$num_jobs_nnet $dir/log/train.$x.JOB.log \
      nnet-shuffle-egs --buffer-size=$shuffle_buffer_size --srand=$x \
      ark:$egs_dir/egs.JOB.$[$x%$iters_per_epoch].ark ark:- \| \
      nnet-train-parallel --num-threads=$num_threads \
         --minibatch-size=$minibatch_size --srand=$x $dir/$x.mdl \
        ark:- $dir/$[$x+1].JOB.mdl \
      || exit 1;

    nnets_list=
    for n in `seq 1 $num_jobs_nnet`; do
      nnets_list="$nnets_list $dir/$[$x+1].$n.mdl"
    done

    learning_rate=`perl -e '($x,$n,$i,$f)=@ARGV; print ($x >= $n ? $f : $i*exp($x*log($f/$i)/$n));' $[$x+1] $num_iters_reduce $initial_learning_rate $final_learning_rate`;
    softmax_learning_rate=`perl -e "print $learning_rate * $softmax_learning_rate_factor;"`;
    nnet-am-info $dir/$[$x+1].1.mdl > $dir/foo  2>/dev/null || exit 1
    nu=`cat $dir/foo | grep num-updatable-components | awk '{print $2}'`
    na=`cat $dir/foo | grep AffineComponent | wc -l` # number of last AffineComopnent layer [one-based]
    lr_string="$learning_rate"
    for n in `seq 2 $nu`; do 
      if [ $n -eq $na ]; then lr=$softmax_learning_rate;
      else lr=$learning_rate; fi
      lr_string="$lr_string:$lr"
    done
    
    $cmd $dir/log/average.$x.log \
      nnet-am-average $nnets_list - \| \
      nnet-am-copy --learning-rates=$lr_string - $dir/$[$x+1].mdl || exit 1;

    rm $nnets_list
  fi
  x=$[$x+1]
done

# Now do combination.
# At the end, final.mdl will be a combination of the last e.g. 10 models.
if [ $num_iters_final -gt $num_iters_extra ]; then
  echo "Setting num_iters_final=$num_iters_extra"
  num_iters_final=$num_iters_extra
fi
start=$[$num_iters-$num_iters_final+1]
nnets_list=
for x in `seq $start $num_iters`; do
  nnets_list="$nnets_list $dir/$x.mdl"
done

if [ $stage -le $num_iters ]; then
  num_egs=`nnet-copy-egs ark:$egs_dir/combine.egs ark:/dev/null 2>&1 | tail -n 1 | awk '{print $NF}'`
  mb=$[($num_egs+$num_threads-1)/$num_threads]
  $cmd $parallel_opts $dir/log/combine.log \
    nnet-combine-fast --use-gpu=no --num-threads=$num_threads --verbose=3 --minibatch-size=$mb \
    $nnets_list ark:$egs_dir/combine.egs $dir/final.mdl || exit 1;
fi

sleep 2; # make sure final.mdl exists.

# Compute the probability of the final, combined model with
# the same subset we used for the previous compute_probs, as the
# different subsets will lead to different probs.
$cmd $dir/log/compute_prob_valid.final.log \
  nnet-compute-prob $dir/final.mdl ark:$egs_dir/valid_diagnostic.egs &
$cmd $dir/log/compute_prob_train.final.log \
  nnet-compute-prob $dir/final.mdl ark:$egs_dir/train_diagnostic.egs &

echo Done

if $cleanup; then
  echo Removing most of the models
  for x in `seq 0 $num_iters`; do
    if [ $[$x%10] -ne 0 ] && [ $x -lt $[$num_iters-$num_iters_final+1] ]; then 
       # delete all but every 10th model; don't delete the ones which combine to form the final model.
      rm $dir/$x.mdl
    fi
  done
fi
