#!/bin/bash

# Copyright 2016 Johns Hopkins University (Author: Daniel Povey).
# Apache 2.0.


# Begin configuration section.
cmd=run.pl
num_epochs=4      # Number of epochs of training;
                  # the number of iterations is worked out from this.
num_shifts=1
initial_effective_lrate=0.0003
final_effective_lrate=0.00003
num_jobs_initial=2 # Number of neural net jobs to run in parallel at the start of training
num_jobs_final=8   # Number of neural net jobs to run in parallel at the end of training
stage=-3
diagnostic_period=5
compute_accuracy=true


shuffle_buffer_size=1000 # This "buffer_size" variable controls randomization of the samples
                # on each iter.  You could set it to 0 or to a large value for complete
                # randomization, but this would both consume memory and cause spikes in
                # disk I/O.  Smaller is easier on disk and memory but less random.  It's
                # not a huge deal though, as samples are anyway randomized right at the start.
                # (the point of this is to get data in different minibatches on different iterations,
                # since in the preconditioning method, 2 samples in the same minibatch can
                # affect each others' gradients.

max_param_change=0.2  # max param change per minibatch to use eventually
                      # (for first epoch we use half this)
minibatch_size=256   # minibatch size to use eventually
                     # (for first epoch we use half this)

use_gpu=true    # if true, we run on GPU.
egs_dir=

# End configuration section.

trap 'for pid in $(jobs -pr); do kill -KILL $pid; done' INT QUIT TERM

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 1 ]; then
  echo "Usage: $0 [opts] <exp-dir>"
  echo " e.g.: $0 data/train data/lang exp/tri3_ali exp/tri4_nnet"
  echo "This script trains the xvector system; see egs/swbd/s5c/local/xvector/train.sh for"
  echo "example (you have to create the nnet configs and the egs first)."
  echo ""
  echo "Main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config file containing options"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --num-epochs <#epochs|10>                        # Number of epochs of training"
  echo "  --initial-effective-lrate <lrate|0.0003>         # effective learning rate at start of training."
  echo "  --final-effective-lrate <lrate|0.00003>          # effective learning rate at end of training."
  echo "                                                   # data, 0.00025 for large data"
  echo "  --num-jobs-initial <num-jobs|1>                  # Number of parallel jobs to use for neural net training, at the start."
  echo "  --num-jobs-final <num-jobs|8>                    # Number of parallel jobs to use for neural net training, at the end"
  echo "  --egs-dir <egs-dir>                              # If supplied, overrides <exp-dir>/egs as location of egs"
  echo "  --stage <stage|-4>                               # Used to run a partially-completed training process from somewhere in"
  echo "                                                   # the middle."
  exit 1;
fi

dir=$1

[ -z $egs_dir ] && egs_dir=$dir/egs

if [ ! -d $egs_dir/info ]; then
  echo "$0: expected $egs_dir/info to exist: did you run steps/nnet3/xvector/get_egs.sh first?"
  exit 1
fi
if [ ! -f $dir/configs/final.config ]; then
  echo "$0: expected $dir/configs/final.config to exist (e.g. run steps/nnet3/xvector/make_jesus_configs.py first)"
  exit 1
fi


num_archives=$(cat $egs_dir/info/num_archives)
num_diagnostic_archives=$(cat $egs_dir/info/num_diagnostic_archives)



[ $num_jobs_initial -gt $num_jobs_final ] && \
  echo "$0: --initial-num-jobs cannot exceed --final-num-jobs" && exit 1;

[ $num_jobs_final -gt $num_archives ] && \
  echo "$0: --final-num-jobs cannot exceed #archives $num_archives." && exit 1;

# set num_iters so that as close as possible, we process the data $num_epochs
# times $num_shifts times, times, i.e. $num_iters*$avg_num_jobs) ==
# $num_epochs*$num_archives*$num_shifts, where
# avg_num_jobs=(num_jobs_initial+num_jobs_final)/2.
num_archives_to_process=$[$num_epochs*$num_archives*$num_shifts]
num_archives_processed=0
num_iters=$[($num_archives_to_process*2)/($num_jobs_initial+$num_jobs_final)]

echo "$0: Will train for $num_epochs epochs = $num_iters iterations"

if $use_gpu; then
  parallel_suffix=""
  train_queue_opt="--gpu 1"
  parallel_train_opts=
  if ! cuda-compiled; then
    echo "$0: WARNING: you are running with one thread but you have not compiled"
    echo "   for CUDA.  You may be running a setup optimized for GPUs.  If you have"
    echo "   GPUs and have nvcc installed, go to src/ and do ./configure; make"
    exit 1
  fi
else
  echo "$0: without using a GPU this will be very slow.  nnet3 does not yet support multiple threads."
  parallel_train_opts="--use-gpu=no"
fi

if [ $stage -le -1 ]; then
  $cmd $dir/log/nnet_init.log \
    nnet3-init $dir/configs/final.config $dir/0.raw || exit 1
fi


x=0

while [ $x -lt $num_iters ]; do

  this_num_jobs=$(perl -e "print int(0.5+$num_jobs_initial+($num_jobs_final-$num_jobs_initial)*$x/$num_iters);")

  ilr=$initial_effective_lrate; flr=$final_effective_lrate; np=$num_archives_processed; nt=$num_archives_to_process;
  this_effective_learning_rate=$(perl -e "print ($x + 1 >= $num_iters ? $flr : $ilr*exp($np*log($flr/$ilr)/$nt));");
  this_learning_rate=$(perl -e "print ($this_effective_learning_rate*$this_num_jobs);");

  if [ $stage -le $x ]; then
    echo "On iteration $x, learning rate is $this_learning_rate"
    raw="nnet3-copy --learning-rate=$this_learning_rate $dir/$x.raw - |"

    if [ $[$x%$diagnostic_period] == 0 ]; then
      # Set off jobs doing some diagnostics, in the background.
      $cmd JOB=1:$num_diagnostic_archives $dir/log/compute_prob_valid.$x.JOB.log \
        nnet3-xvector-compute-prob --compute-accuracy=${compute_accuracy} $dir/$x.raw \
        "ark:nnet3-merge-egs --measure-output-frames=false ark:$egs_dir/valid_diagnostic_egs.JOB.ark ark:- |" &
      $cmd JOB=1:$num_diagnostic_archives $dir/log/compute_prob_train.$x.JOB.log \
        nnet3-xvector-compute-prob --compute-accuracy=${compute_accuracy} $dir/$x.raw \
        "ark:nnet3-merge-egs --measure-output-frames=false ark:$egs_dir/train_diagnostic_egs.JOB.ark ark:- |" &
    fi
    if [ $x -gt 0 ]; then
      $cmd $dir/log/progress.$x.log \
        nnet3-info $dir/$x.raw '&&' \
        nnet3-show-progress --use-gpu=no $dir/$[$x-1].raw $dir/$x.raw &
    fi

    echo "Training neural net (pass $x)"

    if [ $x -le 1 ]; then
      do_average=false # for the first 2 iters, don't do averaging, pick the best.
    else
      do_average=true
    fi

    rm $dir/.error 2>/dev/null


    ( # this sub-shell is so that when we "wait" below,
      # we only wait for the training jobs that we just spawned,
      # not the diagnostic jobs that we spawned above.

      # We cannot easily use a single parallel SGE job to do the main training,
      # because the computation of which archive and which --frame option
      # to use for each job is a little complex, so we spawn each one separately.
      # this is no longer true for RNNs as we use do not use the --frame option
      # but we use the same script for consistency with FF-DNN code

      for n in $(seq $this_num_jobs); do
        k=$[$num_archives_processed + $n - 1]; # k is a zero-based index that we will derive
                                               # the other indexes from.
        archive=$[($k%$num_archives)+1]; # work out the 1-based archive index.
        frame_shift=$[($k/$num_archives)%$num_shifts];

        this_max_param_change=$max_param_change
        this_minibatch_size=$minibatch_size
        # for the first 20 iterations or the first epoch, whichever comes earlier,
        # use a smaller minibatch size and max-param-change.
        if [ $k -lt $[$num_archives*$num_shifts] ] && [ $x -lt 20 ]; then
          # if we're the first epoch, use half the minibatch size and half the
          # max-param-change.
          this_minibatch_size=$[$minibatch_size/2]
          this_max_param_change=$(perl -e "print ($max_param_change / 2.0);")
        fi

        $cmd $train_queue_opt $dir/log/train.$x.$n.log \
          nnet3-xvector-train $parallel_train_opts --print-interval=10 \
          --max-param-change=$this_max_param_change "$raw" \
          "ark:nnet3-copy-egs --frame-shift=$frame_shift ark:$egs_dir/egs.$archive.ark ark:- | nnet3-shuffle-egs --buffer-size=$shuffle_buffer_size --srand=$x ark:- ark:-| nnet3-merge-egs --measure-output-frames=false --minibatch-size=$this_minibatch_size --discard-partial-minibatches=true ark:- ark:- |" \
          $dir/$[$x+1].$n.raw || touch $dir/.error &
      done
      wait
      if [ -f $dir/.error ]; then
        echo "$0: error detected on iteration $x of training"
        exit 1
      fi
    )
    # the error message below is not that informative, but $cmd will
    # have printed a more specific one.
    [ -f $dir/.error ] && echo "$0: error on iteration $x of training" && exit 1;

    models_to_average=$(steps/nnet3/get_successful_models.py $this_num_jobs $dir/log/train.$x.%.log)
    nnets_list=
    for n in $models_to_average; do
      nnets_list="$nnets_list $dir/$[$x+1].$n.raw"
    done

    if $do_average; then
      # average the output of the different jobs.
      $cmd $dir/log/average.$x.log \
        nnet3-average $nnets_list $dir/$[$x+1].raw || exit 1;
    else
      # choose the best from the different jobs.
      n=$(perl -e '($nj,$pat)=@ARGV; $best_n=1; $best_logprob=-1.0e+10; for ($n=1;$n<=$nj;$n++) {
          $fn = sprintf($pat,$n); open(F, "<$fn") || die "Error opening log file $fn";
          undef $logprob; while (<F>) { if (m/log-prob-per-frame=(\S+)/) { $logprob=$1; } }
          close(F); if (defined $logprob && $logprob > $best_logprob) { $best_logprob=$logprob;
          $best_n=$n; } } print "$best_n\n"; ' $this_num_jobs $dir/log/train.$x.%d.log) || exit 1;
      [ -z "$n" ] && echo "Error getting best model" && exit 1;
      $cmd $dir/log/select.$x.log \
        cp $dir/$[$x+1].$n.raw $dir/$[$x+1].raw || exit 1;
    fi

    nnets_list=
    for n in `seq 1 $this_num_jobs`; do
      nnets_list="$nnets_list $dir/$[$x+1].$n.raw"
    done

    rm $nnets_list
    [ ! -f $dir/$[$x+1].raw ] && exit 1;
    if [ -f $dir/$[$x-1].raw ] && $cleanup && \
       [ $[($x-1)%100] -ne 0  ]; then
      rm $dir/$[$x-1].raw
    fi
  fi
  rm $dir/cache.$x 2>/dev/null
  x=$[$x+1]
  num_archives_processed=$[$num_archives_processed+$this_num_jobs]
done


cp $dir/$x.raw $dir/final.raw

# don't bother with combination for now - it makes very little difference.

sleep 2

echo Done
