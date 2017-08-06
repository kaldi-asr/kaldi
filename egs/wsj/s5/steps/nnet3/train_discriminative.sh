#!/bin/bash

# Copyright 2012-2014  Johns Hopkins University (Author: Daniel Povey)
#           2014-2015  Vimal Manohar
# Apache 2.0.

set -o pipefail

# This script does MPE or MMI or state-level minimum bayes risk (sMBR) training
# using egs obtained by steps/nnet3/get_egs_discriminative.sh

# Begin configuration section.
cmd=run.pl
num_epochs=4       # Number of epochs of training;
                   # the number of iterations is worked out from this.
                   # Be careful with this: we actually go over the data
                   # num-epochs * frame-subsampling-factor times, due to
                   # using different data-shifts.
use_gpu=true
apply_deriv_weights=true
use_frame_shift=false
run_diagnostics=true
learning_rate=0.00002
max_param_change=2.0
scale_max_param_change=false # if this option is used, scale it by num-jobs.

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
regularization_opts=
minibatch_size=64  # This is the number of examples rather than the number of output frames.
last_layer_factor=1.0  # relates to modify-learning-rates [deprecated]
shuffle_buffer_size=1000 # This "buffer_size" variable controls randomization of the samples
                # on each iter.  You could set it to 0 or to a large value for complete
                # randomization, but this would both consume memory and cause spikes in
                # disk I/O.  Smaller is easier on disk and memory but less random.  It's
                # not a huge deal though, as samples are anyway randomized right at the start.


stage=-3

num_threads=16  # this is the default but you may want to change it, e.g. to 1 if
                # using GPUs.

cleanup=true
keep_model_iters=100
remove_egs=false
src_model=  # will default to $degs_dir/final.mdl

num_jobs_compute_prior=10

min_deriv_time=0
max_deriv_time_relative=0
# End configuration section.


echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;


if [ $# != 2 ]; then
  echo "Usage: $0 [opts] <degs-dir> <exp-dir>"
  echo " e.g.: $0 exp/nnet3/tdnn_sp_degs exp/nnet3/tdnn_sp_smbr"
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
  echo "                                                   # changes across layers. [deprecated]"
  exit 1;
fi

degs_dir=$1
dir=$2

[ -z "$src_model" ] && src_model=$degs_dir/final.mdl

# Check some files.
for f in $degs_dir/degs.1.ark $degs_dir/info/{num_archives,silence.csl,frame_subsampling_factor} $src_model; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

mkdir -p $dir/log || exit 1;


model_left_context=$(nnet3-am-info $src_model | grep "^left-context:" | awk '{print $2}')
model_right_context=$(nnet3-am-info $src_model | grep "^right-context:" | awk '{print $2}')

# Copy the ivector information
if [ -f $degs_dir/info/final.ie.id ]; then
  cp $degs_dir/info/final.ie.id $dir/ 2>/dev/null || true
fi

# copy some things
for f in splice_opts cmvn_opts tree final.mat; do
  if [ -f $degs_dir/$f ]; then
    cp $degs_dir/$f $dir/ || exit 1;
  fi
done

silphonelist=`cat $degs_dir/info/silence.csl` || exit 1;

num_archives=$(cat $degs_dir/info/num_archives) || exit 1;
frame_subsampling_factor=$(cat $degs_dir/info/frame_subsampling_factor)

echo $frame_subsampling_factor > $dir/frame_subsampling_factor

if $use_frame_shift; then
  num_archives_expanded=$[$num_archives*$frame_subsampling_factor]
else
  num_archives_expanded=$num_archives
fi

if [ $num_jobs_nnet -gt $num_archives_expanded ]; then
  echo "$0: num-jobs-nnet $num_jobs_nnet exceeds number of archives $num_archives_expanded,"
  echo " ... setting it to $num_archives."
  num_jobs_nnet=$num_archives_expanded
fi

num_archives_to_process=$[$num_epochs*$num_archives_expanded]
num_archives_processed=0
num_iters=$[$num_archives_to_process/$num_jobs_nnet]

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

if $use_frame_shift; then
  num_epochs_expanded=$[num_epochs*frame_subsampling_factor]
else
  num_epochs_expanded=$num_epochs
fi

for e in $(seq 1 $num_epochs_expanded); do
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


  # set the learning rate to $learning_rate, and
  # set the output-layer's learning rate to
  # $learning_rate times $last_layer_factor.
  edits_str="set-learning-rate learning-rate=$learning_rate"
  if [ "$last_layer_factor" != "1.0" ]; then
    last_layer_lrate=$(perl -e "print ($learning_rate*$last_layer_factor);") || exit 1
    edits_str="$edits_str; set-learning-rate name=output.affine learning-rate=$last_layer_lrate"
  fi

  $cmd $dir/log/convert.log \
    nnet3-am-copy --edits="$edits_str" "$src_model" $dir/0.mdl || exit 1;

  ln -sf 0.mdl $dir/epoch0.mdl
fi


rm $dir/.error 2>/dev/null

x=0

while [ $x -lt $num_iters ]; do
  if [ $stage -le $x ]; then
    if $run_diagnostics; then
      # Set off jobs doing some diagnostics, in the background.  # Use the egs dir from the previous iteration for the diagnostics
      $cmd $dir/log/compute_objf_valid.$x.log \
        nnet3-discriminative-compute-objf  $regularization_opts \
        --silence-phones=$silphonelist \
        --criterion=$criterion --drop-frames=$drop_frames \
        --one-silence-class=$one_silence_class \
        --boost=$boost --acoustic-scale=$acoustic_scale \
        $dir/$x.mdl \
        ark:$degs_dir/valid_diagnostic.degs &
      $cmd $dir/log/compute_objf_train.$x.log \
        nnet3-discriminative-compute-objf  $regularization_opts \
        --silence-phones=$silphonelist \
        --criterion=$criterion --drop-frames=$drop_frames \
        --one-silence-class=$one_silence_class \
        --boost=$boost --acoustic-scale=$acoustic_scale \
        $dir/$x.mdl \
        ark:$degs_dir/train_diagnostic.degs &
    fi

    if [ $x -gt 0 ]; then
      $cmd $dir/log/progress.$x.log \
        nnet3-show-progress --use-gpu=no "nnet3-am-copy --raw=true $dir/$[$x-1].mdl - |" "nnet3-am-copy --raw=true $dir/$x.mdl - |" \
        '&&' \
        nnet3-info "nnet3-am-copy --raw=true $dir/$x.mdl - |" &
    fi


    echo "Training neural net (pass $x)"

    cache_read_opt="--read-cache=$dir/cache.$x"

    ( # this sub-shell is so that when we "wait" below,
      # we only wait for the training jobs that we just spawned,
      # not the diagnostic jobs that we spawned above.

      # We can't easily use a single parallel SGE job to do the main training,
      # because the computation of which archive and which --frame option
      # to use for each job is a little complex, so we spawn each one separately.
      for n in `seq $num_jobs_nnet`; do
        k=$[$num_archives_processed + $n - 1]; # k is a zero-based index that we'll derive
                                               # the other indexes from.
        archive=$[($k%$num_archives)+1]; # work out the 1-based archive index.

        if [ $n -eq 1 ]; then
          # an option for writing cache (storing pairs of nnet-computations and
          # computation-requests) during training.
          cache_write_opt=" --write-cache=$dir/cache.$[$x+1]"
        else
          cache_write_opt=""
        fi

        if $use_frame_shift; then
          frame_shift=$[(k%num_archives + k/num_archives) % frame_subsampling_factor]
        else
          frame_shift=0
        fi

        #archive=$[(($n+($x*$num_jobs_nnet))%$num_archives)+1]
        if $scale_max_param_change; then
          this_max_param_change=$(perl -e "print ($max_param_change * $num_jobs_nnet);")
        else
          this_max_param_change=$max_param_change
        fi

        $cmd $train_queue_opt $dir/log/train.$x.$n.log \
          nnet3-discriminative-train $cache_read_opt $cache_write_opt \
          --apply-deriv-weights=$apply_deriv_weights \
          --optimization.min-deriv-time=-$model_left_context \
          --optimization.max-deriv-time-relative=$model_right_context \
            $parallel_train_opts \
          --max-param-change=$this_max_param_change \
          --silence-phones=$silphonelist \
          --criterion=$criterion --drop-frames=$drop_frames \
          --one-silence-class=$one_silence_class \
          --boost=$boost --acoustic-scale=$acoustic_scale $regularization_opts \
          $dir/$x.mdl \
          "ark,bg:nnet3-discriminative-copy-egs --frame-shift=$frame_shift ark:$degs_dir/degs.$archive.ark ark:- | nnet3-discriminative-shuffle-egs --buffer-size=$shuffle_buffer_size --srand=$x ark:- ark:- | nnet3-discriminative-merge-egs --minibatch-size=$minibatch_size ark:- ark:- |" \
          $dir/$[$x+1].$n.raw || touch $dir/.error &
      done
      wait
      [ -f $dir/.error ] && exit 1
    )
    [ -f $dir/.error ] && { echo "Found $dir/.error. See $dir/log/train.$x.*.log"; exit 1; }

    nnets_list=$(for n in $(seq $num_jobs_nnet); do echo $dir/$[$x+1].$n.raw; done)

    # below use run.pl instead of a generic $cmd for these very quick stages,
    # so that we don't run the risk of waiting for a possibly hard-to-get GPU.
    run.pl $dir/log/average.$x.log \
      nnet3-average $nnets_list - \| \
      nnet3-am-copy --set-raw-nnet=- $dir/$x.mdl $dir/$[$x+1].mdl || exit 1;

    rm $nnets_list
    [ ! -f $dir/$[$x+1].mdl ] && echo "$0: Did not create $dir/$[$x+1].mdl" && exit 1;
    if [ -f $dir/$[$x-1].mdl ] && $cleanup && \
       [ $[($x-1)%$keep_model_iters] -ne 0  ] && \
       [ -z "${iter_to_epoch[$[$x-1]]}" ]; then
      rm $dir/$[$x-1].mdl
    fi

    [ -f $dir/.error ] && { echo "Found $dir/.error. Error on iteration $x"; exit 1; }
  fi

  rm $dir/cache.$x 2>/dev/null || true
  x=$[$x+1]
  num_archives_processed=$[num_archives_processed+num_jobs_nnet]

  if [ $stage -le $x ] && [ ! -z "${iter_to_epoch[$x]}" ]; then
    e=${iter_to_epoch[$x]}
    ln -sf $x.mdl $dir/epoch$e.mdl

    (
      rm $dir/.error 2> /dev/null

      steps/nnet3/adjust_priors.sh --egs-type degs \
        --num-jobs-compute-prior $num_jobs_compute_prior \
        --cmd "$cmd" --use-gpu false \
        --minibatch-size $minibatch_size \
        --use-raw-nnet false --iter epoch$e $dir $degs_dir \
        || { touch $dir/.error; echo "Error in adjusting priors. See errors above."; exit 1; }
    ) &
  fi

done

rm $dir/final.mdl 2>/dev/null
cp $dir/$x.mdl $dir/final.mdl

# function to remove egs that might be soft links.
remove () { for x in $*; do [ -L $x ] && rm $(utils/make_absolute.sh $x); rm $x; done }

if $cleanup && $remove_egs; then  # note: this is false by default.
  echo Removing training examples
  remove $degs_dir/degs.*
  remove $degs_dir/priors_egs.*
fi


if $cleanup; then
  echo Removing most of the models
  for x in `seq 1 $keep_model_iters $num_iters`; do
    if [ -z "${iter_to_epoch[$x]}" ]; then
      # if $x is not an epoch-final iteration..
      rm $dir/$x.mdl 2>/dev/null
    fi
  done
fi

wait
[ -f $dir/.error ] && { echo "Found $dir/.error."; exit 1; }

echo Done && exit 0
