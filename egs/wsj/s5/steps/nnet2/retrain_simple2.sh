#!/bin/bash

# Copyright 2012-2014  Johns Hopkins University (Author: Daniel Povey). 
#                2013  Xiaohui Zhang
#                2013  Guoguo Chen
#                2014  Vimal Manohar
# Apache 2.0.


# retrain_simple2.sh is a script for training a single-layer (softmax-only)
# neural network on top of activations dumped from an existing network; we'll
# later combine the networks to a single network.

# It differs from train_pnorm_simple2.sh in the same way that retrain_fast.sh
# differs from train_pnorm_fast.sh.


# Begin configuration section.
cmd=run.pl
num_epochs=5      # Number of epochs of training;
initial_learning_rate=0.04
final_learning_rate=0.004
minibatch_size=128 # by default use a smallish minibatch size for neural net
                   # training; this controls instability which would otherwise
                   # be a problem with multi-threaded update. 
samples_per_iter=400000 # each iteration of training, see this many samples
                        # per job.  This option is passed to get_egs.sh
num_jobs_nnet=4    # Number of neural net jobs to run in parallel.  This option
                   # is passed to get_egs.sh.
prior_subset_size=10000 # 10k samples per job, for computing priors.  Should be
                        # more than enough.
num_jobs_compute_prior=10 # these are single-threaded, run on CPU.
get_egs_stage=0


max_models_combine=20 # The "max_models_combine" is the maximum number of models we give
  # to the final 'combine' stage, but these models will themselves be averages of
  # iteration-number ranges.

shuffle_buffer_size=5000 # This "buffer_size" variable controls randomization of the samples
                # on each iter.  You could set it to 0 or to a large value for complete
                # randomization, but this would both consume memory and cause spikes in
                # disk I/O.  Smaller is easier on disk and memory but less random.  It's
                # not a huge deal though, as samples are anyway randomized right at the start.
                # (the point of this is to get data in different minibatches on different iterations,
                # since in the preconditioning method, 2 samples in the same minibatch can
                # affect each others' gradients.

stage=-4


alpha=4.0 # relates to online preconditioning.
update_period=4 # relates to online preconditioning: says how often we update the subspace.
num_samples_history=2000 # relates to online preconditioning
precondition_rank_in=20  # relates to online preconditioning
precondition_rank_out=80 # relates to online preconditioning
max_change_per_sample=0.075

mix_up=0 # Number of components to mix up to (should be > #tree leaves, if
        # specified.)
num_threads=16
parallel_opts="-pe smp 16 -l ram_free=1G,mem_free=1G" 
  # by default we use 16 threads; this lets the queue know.
  # note: parallel_opts doesn't automatically get adjusted if you adjust num-threads.
combine_num_threads=8
combine_parallel_opts="-pe smp 8"  # queue options for the "combine" stage.
cleanup=true
egs_dir=
egs_opts=
io_opts="-tc 5" # for jobs with a lot of I/O, limits the number running at one time.
align_cmd=              # The cmd that is passed to steps/nnet2/align.sh
align_use_gpu=          # Passed to use_gpu in steps/nnet2/align.sh [yes/no]
realign_epochs=         # List of epochs, the beginning of which realignment is done
num_jobs_align=30       # Number of jobs for realignment
# End configuration section.


echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 4 ]; then
  echo "Usage: $0 [opts] <data> <lang> <ali-dir> <exp-dir>"
  echo " e.g.: $0 data/train data/lang exp/tri3_ali exp/tri4_nnet"
  echo ""
  echo "Main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config file containing options"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --num-epochs <#epochs|5>                         # Number of epochs of training"
  echo "  --initial-learning-rate <initial-learning-rate|0.02> # Learning rate at start of training, e.g. 0.02 for small"
  echo "                                                       # data, 0.01 for large data"
  echo "  --final-learning-rate  <final-learning-rate|0.004>   # Learning rate at end of training, e.g. 0.004 for small"
  echo "                                                   # data, 0.001 for large data"
  echo "  --mix-up <#pseudo-gaussians|0>                   # Can be used to have multiple targets in final output layer,"
  echo "                                                   # per context-dependent state.  Try a number several times #states."
  echo "  --num-jobs-nnet <num-jobs|8>                     # Number of parallel jobs to use for main neural net"
  echo "                                                   # training (will affect results as well as speed; try 8, 16)"
  echo "                                                   # Note: if you increase this, you may want to also increase"
  echo "                                                   # the learning rate."
  echo "  --num-threads <num-threads|16>                   # Number of parallel threads per job (will affect results"
  echo "                                                   # as well as speed; may interact with batch size; if you increase"
  echo "                                                   # this, you may want to decrease the batch size."
  echo "  --parallel-opts <opts|\"-pe smp 16 -l ram_free=1G,mem_free=1G\">      # extra options to pass to e.g. queue.pl for processes that"
  echo "                                                   # use multiple threads... note, you might have to reduce mem_free,ram_free"
  echo "                                                   # versus your defaults, because it gets multiplied by the -pe smp argument."
  echo "  --io-opts <opts|\"-tc 10\">                      # Options given to e.g. queue.pl for jobs that do a lot of I/O."
  echo "  --minibatch-size <minibatch-size|128>            # Size of minibatch to process (note: product with --num-threads"
  echo "                                                   # should not get too large, e.g. >2k)."
  echo "  --samples-per-iter <#samples|400000>             # Number of samples of data to process per iteration, per"
  echo "                                                   # process."
  echo "  --realign-epochs <list-of-epochs|\"\">           # A list of space-separated epoch indices the beginning of which"
  echo "                                                   # realignment is to be done"
  echo "  --align-cmd (utils/run.pl|utils/queue.pl <queue opts>) # passed to align.sh"
  echo "  --align-use-gpu (yes/no)                         # specify is gpu is to be used for realignment"
  echo "  --num-jobs-align <#njobs|30>                     # Number of jobs to perform realignment"
  echo "  --stage <stage|-4>                               # Used to run a partially-completed training process from somewhere in"
  echo "                                                   # the middle."

  
  exit 1;
fi

data=$1
lang=$2
alidir=$3
dir=$4

if [ ! -z "$realign_epochs" ]; then
  [ -z "$align_cmd" ] && echo "$0: realign_epochs specified but align_cmd not specified" && exit 1
  [ -z "$align_use_gpu" ] && echo "$0: realign_epochs specified but align_use_gpu not specified" && exit 1
fi

# Check some files.
for f in $data/feats.scp $lang/L.fst $alidir/ali.1.gz $alidir/final.mdl $alidir/tree; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done


# Set some variables.
num_leaves=`tree-info $alidir/tree 2>/dev/null | grep num-pdfs | awk '{print $2}'` || exit 1
[ -z $num_leaves ] && echo "\$num_leaves is unset" && exit 1
[ "$num_leaves" -eq "0" ] && echo "\$num_leaves is 0" && exit 1

nj=`cat $alidir/num_jobs` || exit 1;  # number of jobs in alignment dir...
# in this dir we'll have just one job.
sdata=$data/split$nj
utils/split_data.sh $data $nj

mkdir -p $dir/log
echo $nj > $dir/num_jobs
cp $alidir/tree $dir



if [ $stage -le -3 ] && [ -z "$egs_dir" ]; then
  echo "$0: calling get_egs2.sh"
  # set --frames-per-eg to 1 because there is no context, so there
  # is no advantage in having multiple frames per eg.
  steps/nnet2/get_egs2.sh --feat-type raw \
    --frames-per-eg 1 --left-context 0 --right-context 0 \
    --cmvn-opts "--norm-means=false --norm-vars=false" \
    --io-opts "$io_opts" \
    --samples-per-iter $samples_per_iter --stage $get_egs_stage \
    --cmd "$cmd" $egs_opts $data $alidir $dir/egs || exit 1;
fi

if [ -z $egs_dir ]; then
  egs_dir=$dir/egs
fi

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

if [ $stage -le -2 ]; then
  echo "$0: initializing neural net";
  feat_dim=$(feat-to-dim scp:$data/feats.scp -) || exit 1;

  online_preconditioning_opts="alpha=$alpha num-samples-history=$num_samples_history update-period=$update_period rank-in=$precondition_rank_in rank-out=$precondition_rank_out max-change-per-sample=$max_change_per_sample"

  stddev=`perl -e "print 1.0/sqrt($pnorm_input_dim);"`
  cat >$dir/nnet.config <<EOF
AffineComponentPreconditionedOnline input-dim=$feat_dim output-dim=$num_leaves $online_preconditioning_opts learning-rate=$initial_learning_rate param-stddev=0 bias-stddev=0
SoftmaxComponent dim=$num_leaves
EOF

  $cmd $dir/log/nnet_init.log \
    nnet-am-init $alidir/tree $lang/topo "nnet-init $dir/nnet.config -|" \
    $dir/0.mdl || exit 1;
fi

if [ $stage -le -1 ]; then
  echo "Training transition probabilities and setting priors"
  $cmd $dir/log/train_trans.log \
    nnet-train-transitions $dir/0.mdl "ark:gunzip -c $alidir/ali.*.gz|" $dir/0.mdl \
    || exit 1;
fi

# set num_iters so that as close as possible, we process the data $num_epochs
# times, i.e. $num_iters*$num_jobs_nnet == $num_epochs*$num_archives_expanded
num_iters=$[($num_epochs*$num_archives_expanded)/$num_jobs_nnet]

echo "$0: Will train for $num_epochs epochs = $num_iters iterations"

mix_up_iter=$[$num_iters/2]

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

x=0

for realign_epoch in $realign_epochs; do
  # compare the equation below with the equation we use to set num_iters above.
  # note, realign_epochs may be floating-point, which is why we don't use $[] to
  # do the math.
  realign_iter=$(perl -e 'print int(($ARGV[0]*$ARGV[1])/$ARGV[2]);' $realign_epoch $num_archives_expanded $num_jobs_nnet)
  realign_this_iter[$realign_iter]=$realign_epoch
done

cur_egs_dir=$egs_dir

while [ $x -lt $num_iters ]; do
    
  if [ ! -z "${realign_this_iter[$x]}" ]; then
    prev_egs_dir=$cur_egs_dir
    cur_egs_dir=$dir/egs_${realign_this_iter[$x]}
  fi

  if [ $x -ge 0 ] && [ $stage -le $x ]; then
    if [ ! -z "${realign_this_iter[$x]}" ]; then
      epoch=${realign_this_iter[$x]}

             

      echo "Getting average posterior for purposes of adjusting the priors."
      # Note: this just uses CPUs, using a smallish subset of data.
      # always use the first egs archive, which makes the script simpler;
      # we're using different random subsets of it.
      rm $dir/post.$x.*.vec 2>/dev/null
      $cmd JOB=1:$num_jobs_compute_prior $dir/log/get_post.$x.JOB.log \
        nnet-copy-egs --srand=JOB --frame=random ark:$prev_egs_dir/egs.1.ark ark:- \| \
        nnet-subset-egs --srand=JOB --n=$prior_subset_size ark:- ark:- \| \
        nnet-compute-from-egs "nnet-to-raw-nnet $dir/$x.mdl -|" ark:- ark:- \| \
        matrix-sum-rows ark:- ark:- \| vector-sum ark:- $dir/post.$x.JOB.vec || exit 1;

      sleep 3;  # make sure there is time for $dir/post.$x.*.vec to appear.

      $cmd $dir/log/vector_sum.$x.log \
        vector-sum $dir/post.$x.*.vec $dir/post.$x.vec || exit 1;
      rm $dir/post.$x.*.vec;

      echo "Re-adjusting priors based on computed posteriors"
      $cmd $dir/log/adjust_priors.$x.log \
        nnet-adjust-priors $dir/$x.mdl $dir/post.$x.vec $dir/$x.mdl || exit 1;

      sleep 2

      steps/nnet2/align.sh --nj $num_jobs_align --cmd "$align_cmd" --use-gpu $align_use_gpu \
        --transform-dir "$transform_dir" --online-ivector-dir "$online_ivector_dir" \
        --iter $x $data $lang $dir $dir/ali_$epoch || exit 1

      steps/nnet2/relabel_egs2.sh --cmd "$cmd" --iter $x $dir/ali_$epoch \
        $prev_egs_dir $cur_egs_dir || exit 1

      if $cleanup && [[ $prev_egs_dir =~ $dir/egs* ]]; then
        steps/nnet2/remove_egs.sh $prev_egs_dir
      fi
    fi
    
    # Set off jobs doing some diagnostics, in the background.
    # Use the egs dir from the previous iteration for the diagnostics
    $cmd $dir/log/compute_prob_valid.$x.log \
      nnet-compute-prob $dir/$x.mdl ark:$cur_egs_dir/valid_diagnostic.egs &
    $cmd $dir/log/compute_prob_train.$x.log \
      nnet-compute-prob $dir/$x.mdl ark:$cur_egs_dir/train_diagnostic.egs &
    if [ $x -gt 0 ] && [ ! -f $dir/log/mix_up.$[$x-1].log ]; then
      $cmd $dir/log/progress.$x.log \
        nnet-show-progress --use-gpu=no $dir/$[$x-1].mdl $dir/$x.mdl \
        ark:$cur_egs_dir/train_diagnostic.egs '&&' \
        nnet-am-info $dir/$x.mdl &
    fi

    echo "Training neural net (pass $x)"

    mdl=$dir/$x.mdl

    if [ $x -eq 0 ]; then
      # on iteration zero, use a smaller minibatch
      # size and just one job: the model-averaging doesn't seem to be helpful
      # when the model is changing too fast (i.e. it worsens the objective
      # function), and the smaller minibatch size will help to keep
      # the update stable.
      this_minibatch_size=$[$minibatch_size/2];
      do_average=false
    else
      this_minibatch_size=$minibatch_size
      do_average=true
    fi

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
          --minibatch-size=$this_minibatch_size --srand=$x $dir/$x.mdl \
          "ark:nnet-copy-egs --frame=$frame ark:$cur_egs_dir/egs.$archive.ark ark:-|nnet-shuffle-egs --buffer-size=$shuffle_buffer_size --srand=$x ark:- ark:-|" \
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

    learning_rate=`perl -e '($x,$n,$i,$f)=@ARGV; print ($x >= $n ? $f : $i*exp($x*log($f/$i)/$n));' $[$x+1] $num_iters $initial_learning_rate $final_learning_rate`;

    if $do_average; then
      # average the output of the different jobs.
      $cmd $dir/log/average.$x.log \
        nnet-am-average $nnets_list - \| \
        nnet-am-copy --learning-rate=$learning_rate - $dir/$[$x+1].mdl || exit 1;
    else
      # choose the best from the different jobs.
      n=$(perl -e '($nj,$pat)=@ARGV; $best_n=1; $best_logprob=-1.0e+10; for ($n=1;$n<=$nj;$n++) {
          $fn = sprintf($pat,$n); open(F, "<$fn") || die "Error opening log file $fn";
          undef $logprob; while (<F>) { if (m/log-prob-per-frame=(\S+)/) { $logprob=$1; } }
          close(F); if (defined $logprob && $logprob > $best_logprob) { $best_logprob=$logprob; 
          $best_n=$n; } } print "$best_n\n"; ' $num_jobs_nnet $dir/log/train.$x.%d.log) || exit 1;
      [ -z "$n" ] && echo "Error getting best model" && exit 1;
      $cmd $dir/log/select.$x.log \
        nnet-am-copy --learning-rate=$learning_rate $dir/$[$x+1].$n.mdl $dir/$[$x+1].mdl || exit 1;
    fi

    if [ "$mix_up" -gt 0 ] && [ $x -eq $mix_up_iter ]; then
      # mix up.
      echo Mixing up from $num_leaves to $mix_up components
      $cmd $dir/log/mix_up.$x.log \
        nnet-am-mixup --min-count=10 --num-mixtures=$mix_up \
        $dir/$[$x+1].mdl $dir/$[$x+1].mdl || exit 1;
    fi
    rm $nnets_list
    [ ! -f $dir/$[$x+1].mdl ] && exit 1;
    if [ -f $dir/$[$x-1].mdl ] && $cleanup && \
       [ $[($x-1)%100] -ne 0  ] && [ $[$x-1] -lt $first_model_combine ]; then
      rm $dir/$[$x-1].mdl
    fi
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
  num_egs=`nnet-copy-egs ark:$cur_egs_dir/combine.egs ark:/dev/null 2>&1 | tail -n 1 | awk '{print $NF}'`
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
      --verbose=3 --minibatch-size=$mb "${nnets_list[@]}" ark:$cur_egs_dir/combine.egs \
      $dir/final.mdl || exit 1;

  # Compute the probability of the final, combined model with
  # the same subset we used for the previous compute_probs, as the
  # different subsets will lead to different probs.
  $cmd $dir/log/compute_prob_valid.final.log \
    nnet-compute-prob $dir/final.mdl ark:$cur_egs_dir/valid_diagnostic.egs &
  $cmd $dir/log/compute_prob_train.final.log \
    nnet-compute-prob $dir/final.mdl ark:$cur_egs_dir/train_diagnostic.egs &
fi

if [ $stage -le $[$num_iters+1] ]; then
  echo "Getting average posterior for purposes of adjusting the priors."
  # Note: this just uses CPUs, using a smallish subset of data.
  rm $dir/post.$x.*.vec 2>/dev/null
  $cmd JOB=1:$num_jobs_compute_prior $dir/log/get_post.$x.JOB.log \
    nnet-copy-egs --frame=random --srand=JOB ark:$cur_egs_dir/egs.1.ark ark:- \| \
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
  if [[ $cur_egs_dir =~ $dir/egs* ]]; then
    steps/nnet2/remove_egs.sh $cur_egs_dir
  fi

  echo Removing most of the models
  for x in `seq 0 $num_iters`; do
    if [ $[$x%100] -ne 0 ] && [ $x -ne $num_iters ] && [ -f $dir/$x.mdl ]; then
       # delete all but every 100th model; don't delete the ones which combine to form the final model.
      rm $dir/$x.mdl
    fi
  done
fi
