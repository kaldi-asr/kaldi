#!/bin/bash

# Copyright 2012-2014  Johns Hopkins University (Author: Daniel Povey).  Apache 2.0.

# This script does MPE or MMI or state-level minimum bayes risk (sMBR) training.
# This version (2) of the script uses a newer format for the discriminative-training
# egs, as obtained by steps/nnet2/get_egs_discriminative2.sh.

# Begin configuration section.
cmd=run.pl
num_epochs=4       # Number of epochs of training
learning_rate=0.00002
effective_lrate=    # If supplied, overrides the learning rate, which gets set to effective_lrate * num_jobs_nnet.
acoustic_scale=0.1  # acoustic scale for MMI/MPFE/SMBR training.
boost=0.0       # option relevant for MMI

criterion=smbr
drop_frames=false #  option relevant for MMI
one_silence_class=true # option relevant for MPE/SMBR
num_jobs_nnet=4    # Number of neural net jobs to run in parallel.  Note: this
                   # will interact with the learning rates (if you decrease
                   # this, you'll have to decrease the learning rate, and vice
                   # versa).

modify_learning_rates=true
last_layer_factor=1.0  # relates to modify-learning-rates
first_layer_factor=1.0 # relates to modify-learning-rates
shuffle_buffer_size=5000 # This "buffer_size" variable controls randomization of the samples
                # on each iter.  You could set it to 0 or to a large value for complete
                # randomization, but this would both consume memory and cause spikes in
                # disk I/O.  Smaller is easier on disk and memory but less random.  It's
                # not a huge deal though, as samples are anyway randomized right at the start.


stage=-3

adjust_priors=false
num_threads=16  # this is the default but you may want to change it, e.g. to 1 if
                # using GPUs.
parallel_opts="--num-threads 16 --mem 1G"
  # by default we use 16 threads; this lets the queue know.
  # note: parallel_opts doesn't automatically get adjusted if you adjust num-threads.

cleanup=true
retroactive=false
remove_egs=false
src_model=  # will default to $degs_dir/final.mdl
# End configuration section.


echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;


if [ $# != 2 ]; then
  echo "Usage: $0 [opts] <degs-dir> <exp-dir>"
  echo " e.g.: $0 exp/tri4_mpe_degs exp/tri4_mpe"
  echo ""
  echo "You have to first call get_egs_discriminative2.sh to dump the egs."
  echo "Caution: the options 'drop-frames' and 'criterion' are taken here"
  echo "even though they were required also by get_egs_discriminative2.sh,"
  echo "and they should normally match."
  echo ""
  echo "Main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config file containing options"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --num-epochs <#epochs|4>                        # Number of epochs of training"
  echo "  --learning-rate <learning-rate|0.0002>           # Learning rate to use"
  echo "  --effective-lrate <effective-learning-rate>      # If supplied, learning rate will be set to"
  echo "                                                   # this value times num-jobs-nnet."
  echo "  --num-jobs-nnet <num-jobs|8>                     # Number of parallel jobs to use for main neural net"
  echo "                                                   # training (will affect results as well as speed; try 8, 16)"
  echo "                                                   # Note: if you increase this, you may want to also increase"
  echo "                                                   # the learning rate.  Also note: if there are fewer archives"
  echo "                                                   # of egs than this, it will get reduced automatically."
  echo "  --num-threads <num-threads|16>                   # Number of parallel threads per job (will affect results"
  echo "                                                   # as well as speed; may interact with batch size; if you increase"
  echo "                                                   # this, you may want to decrease the batch size.  With GPU, must be 1."
  echo "  --parallel-opts <opts|\"--num-threads 16 --mem 1G\">      # extra options to pass to e.g. queue.pl for processes that"
  echo "                                                   # use multiple threads... "
  echo "  --stage <stage|-3>                               # Used to run a partially-completed training process from somewhere in"
  echo "                                                   # the middle."
  echo "  --criterion <criterion|smbr>                     # Training criterion: may be smbr, mmi or mpfe"
  echo "  --boost <boost|0.0>                              # Boosting factor for MMI (e.g., 0.1)"
  echo "  --drop-frames <true,false|false>                 # Option that affects MMI training: if true, we exclude gradients from frames"
  echo "                                                   # where the numerator transition-id is not in the denominator lattice."
  echo "  --one-silence-class <true,false|false>           # Option that affects MPE/SMBR training (will tend to reduce insertions)"
  echo "  --modify-learning-rates <true,false|false>       # If true, modify learning rates to try to equalize relative"
  echo "                                                   # changes across layers."
  exit 1;
fi

degs_dir=$1
dir=$2

[ -z "$src_model" ] && src_model=$degs_dir/final.mdl

# Check some files.
for f in $degs_dir/degs.1.ark $degs_dir/info/{num_archives,silence.csl,frames_per_archive} $src_model; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

mkdir -p $dir/log || exit 1;

cp $degs_dir/phones.txt $dir 2>/dev/null
# copy some things
for f in splice_opts cmvn_opts tree final.mat; do
  if [ -f $degs_dir/$f ]; then
    cp $degs_dir/$f $dir/ || exit 1;
  fi
done

silphonelist=`cat $degs_dir/info/silence.csl` || exit 1;


num_archives=$(cat $degs_dir/info/num_archives) || exit 1;

if [ $num_jobs_nnet -gt $num_archives ]; then
  echo "$0: num-jobs-nnet $num_jobs_nnet exceeds number of archives $num_archives,"
  echo " ... setting it to $num_archives."
  num_jobs_nnet=$num_archives
fi

num_iters=$[($num_epochs*$num_archives)/$num_jobs_nnet]

echo "$0: Will train for $num_epochs epochs = $num_iters iterations"

for e in $(seq 1 $num_epochs); do
  x=$[($e*$num_archives)/$num_jobs_nnet] # gives the iteration number.
  iter_to_epoch[$x]=$e
done

if [ $stage -le -1 ]; then
  echo "$0: Copying initial model and modifying preconditioning setup"

  # Note, the baseline model probably had preconditioning, and we'll keep it;
  # but we want online preconditioning with a larger number of samples of
  # history, since in this setup the frames are only randomized at the segment
  # level so they are highly correlated.  It might make sense to tune this a
  # little, later on, although I doubt it matters once the --num-samples-history
  # is large enough.

  if [ ! -z "$effective_lrate" ]; then
    learning_rate=$(perl -e "print ($num_jobs_nnet*$effective_lrate);")
    echo "$0: setting learning rate to $learning_rate = --num-jobs-nnet * --effective-lrate."
  fi

  $cmd $dir/log/convert.log \
    nnet-am-copy --learning-rate=$learning_rate "$src_model" - \| \
    nnet-am-switch-preconditioning  --num-samples-history=50000 - $dir/0.mdl || exit 1;
fi



if [ $num_threads -eq 1 ]; then
 train_suffix="-simple" # this enables us to use GPU code if
                        # we have just one thread.
else
  train_suffix="-parallel --num-threads=$num_threads"
fi

rm $dir/.error
x=0   
while [ $x -lt $num_iters ]; do
  if [ $stage -le $x ]; then
    
    echo "Training neural net (pass $x)"

    # The \$ below delays the evaluation of the expression until the script runs (and JOB
    # will be replaced by the job-id).  That expression in $[..] is responsible for
    # choosing the archive indexes to use for each job on each iteration... we cycle through
    # all archives.

    $cmd $parallel_opts JOB=1:$num_jobs_nnet $dir/log/train.$x.JOB.log \
      nnet-combine-egs-discriminative \
        "ark:$degs_dir/degs.\$[((JOB-1+($x*$num_jobs_nnet))%$num_archives)+1].ark" ark:- \| \
      nnet-train-discriminative$train_suffix --silence-phones=$silphonelist \
       --criterion=$criterion --drop-frames=$drop_frames \
       --one-silence-class=$one_silence_class \
       --boost=$boost --acoustic-scale=$acoustic_scale \
       $dir/$x.mdl ark:- $dir/$[$x+1].JOB.mdl || exit 1;

    nnets_list=$(for n in $(seq $num_jobs_nnet); do echo $dir/$[$x+1].$n.mdl; done)

    # below use run.pl instead of a generic $cmd for these very quick stages,
    # so that we don't run the risk of waiting for a possibly hard-to-get GPU.
    run.pl $dir/log/average.$x.log \
      nnet-am-average $nnets_list $dir/$[$x+1].mdl || exit 1;

    if $modify_learning_rates; then
      run.pl $dir/log/modify_learning_rates.$x.log \
        nnet-modify-learning-rates --retroactive=$retroactive \
        --last-layer-factor=$last_layer_factor \
        --first-layer-factor=$first_layer_factor \
        $dir/$x.mdl $dir/$[$x+1].mdl $dir/$[$x+1].mdl || exit 1;
    fi
    rm $nnets_list
  fi
  if $adjust_priors && [ ! -z "${iter_to_epoch[$x]}" ]; then
    if [ ! -f $degs_dir/priors_egs.1.ark ]; then
      echo "$0: Expecting $degs_dir/priors_egs.1.ark to exist since --adjust-priors was true."
      echo "$0: Run this script with --adjust-priors false to not adjust priors"
      exit 1
    fi
    (
    e=${iter_to_epoch[$x]}
    rm $dir/.error
    num_archives_priors=`cat $degs_dir/info/num_archives_priors` || { touch $dir/.error; echo "Could not find $degs_dir/info/num_archives_priors. Set --adjust-priors false to not adjust priors"; exit 1; }

    $cmd JOB=1:$num_archives_priors $dir/log/get_post.epoch$e.JOB.log \
      nnet-compute-from-egs "nnet-to-raw-nnet $dir/$x.mdl -|" \
      ark:$degs_dir/priors_egs.JOB.ark ark:- \| \
      matrix-sum-rows ark:- ark:- \| \
      vector-sum ark:- $dir/post.epoch$e.JOB.vec || \
      { touch $dir/.error; echo "Error in getting posteriors for adjusting priors. See $dir/log/get_post.epoch$e.*.log"; exit 1; }

    sleep 3;

    $cmd $dir/log/sum_post.epoch$e.log \
      vector-sum $dir/post.epoch$e.*.vec $dir/post.epoch$e.vec || \
      { touch $dir/.error; echo "Error in summing posteriors. See $dir/log/sum_post.epoch$e.log"; exit 1; }

    rm $dir/post.epoch$e.*.vec

    echo "Re-adjusting priors based on computed posteriors for iter $x"
    $cmd $dir/log/adjust_priors.epoch$e.log \
      nnet-adjust-priors $dir/$x.mdl $dir/post.epoch$e.vec $dir/$x.mdl \
      || { touch $dir/.error; echo "Error in adjusting priors. See $dir/log/adjust_priors.epoch$e.log"; exit 1; }
    ) &
  fi

  [ -f $dir/.error ] && exit 1

  x=$[$x+1]
done

rm $dir/final.mdl 2>/dev/null
ln -s $x.mdl $dir/final.mdl

echo Done

epoch_final_iters=
for e in $(seq 0 $num_epochs); do
  x=$[($e*$num_archives)/$num_jobs_nnet] # gives the iteration number.
  ln -sf $x.mdl $dir/epoch$e.mdl
  epoch_final_iters="$epoch_final_iters $x"
done


# function to remove egs that might be soft links.
remove () { for x in $*; do [ -L $x ] && rm $(utils/make_absolute.sh $x); rm $x; done }

if $cleanup && $remove_egs; then  # note: this is false by default.
  echo Removing training examples
  for n in $(seq $num_archives); do
    remove $degs_dir/degs.*
    remove $degs_dir/priors_egs.*
  done
fi


if $cleanup; then
  echo Removing most of the models
  for x in `seq 0 $num_iters`; do
    if ! echo $epoch_final_iters | grep -w $x >/dev/null; then 
      # if $x is not an epoch-final iteration..
      rm $dir/$x.mdl 2>/dev/null
    fi
  done
fi

