#!/bin/bash

# Copyright 2012-2014  Johns Hopkins University (Author: Daniel Povey). 
#           2013  Xiaohui Zhang
#           2013  Guoguo Chen
# Apache 2.0.


# train_pnorm_fast.sh is a new, improved version of train_pnorm.sh, which uses
# the 'online' preconditioning method.  For GPUs it's about two times faster
# than before (although that's partly due to optimizations that will also help
# the old recipe), and for CPUs it gives better performance than the old method
# (I believe); also, the difference in optimization performance between CPU and
# GPU is almost gone.  The old train_pnorm.sh script is now deprecated.
# We made this a separate script because not all of the options that the
# old script accepted, are still accepted.

# Begin configuration section.
cmd=run.pl
num_epochs=15      # Number of epochs during which we reduce
                   # the learning rate; number of iterations is worked out from this.
num_epochs_extra=5 # Number of epochs after we stop reducing
                   # the learning rate.
num_iters_final=20 # Maximum number of final iterations to give to the
                   # optimization over the validation set (maximum)
initial_learning_rate=0.04
final_learning_rate=0.004
bias_stddev=0.5
pnorm_input_dim=3000 
pnorm_output_dim=300
p=2
minibatch_size=128 # by default use a smallish minibatch size for neural net
                   # training; this controls instability which would otherwise
                   # be a problem with multi-threaded update. 

samples_per_iter=200000 # each iteration of training, see this many samples
                        # per job.  This option is passed to get_egs.sh
num_jobs_nnet=16   # Number of neural net jobs to run in parallel.  This option
                   # is passed to get_egs.sh.
get_egs_stage=0
online_ivector_dir=

shuffle_buffer_size=5000 # This "buffer_size" variable controls randomization of the samples
                # on each iter.  You could set it to 0 or to a large value for complete
                # randomization, but this would both consume memory and cause spikes in
                # disk I/O.  Smaller is easier on disk and memory but less random.  It's
                # not a huge deal though, as samples are anyway randomized right at the start.
                # (the point of this is to get data in different minibatches on different iterations,
                # since in the preconditioning method, 2 samples in the same minibatch can
                # affect each others' gradients.

add_layers_period=2 # by default, add new layers every 2 iterations.
num_hidden_layers=3
stage=-5

io_opts="-tc 5" # for jobs with a lot of I/O, limits the number running at one time.   These don't
splice_width=4 # meaning +- 4 frames on each side for second LDA
randprune=4.0 # speeds up LDA.
alpha=4.0 # relates to preconditioning.
update_period=4 # relates to online preconditioning: says how often we update the subspace.
num_samples_history=2000 # relates to online preconditioning
max_change_per_sample=0.075
precondition_rank_in=20  # relates to online preconditioning
precondition_rank_out=80 # relates to online preconditioning

# this relates to perturbed training.
min_target_objf_change=0.1
target_multiplier=0 #  Set this to e.g. 1.0 to enable perturbed training.

mix_up=0 # Number of components to mix up to (should be > #tree leaves, if
        # specified.)
num_threads=16
parallel_opts="-pe smp 16 -l ram_free=1G,mem_free=1G" # by default we use 16 threads; this lets the queue know.
  # note: parallel_opts doesn't automatically get adjusted if you adjust num-threads.
combine_num_threads=8
combine_parallel_opts="-pe smp 8"  # queue options for the "combine" stage.
cleanup=true
egs_dir=
lda_opts=
lda_dim=
egs_opts=
transform_dir=     # If supplied, overrides alidir
cmvn_opts=  # will be passed to get_lda.sh and get_egs.sh, if supplied.  
            # only relevant for "raw" features, not lda.
feat_type=  # Can be used to force "raw" features.
prior_subset_size=10000 # 10k samples per job, for computing priors.  Should be
                        # more than enough.
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
  echo "  --num-epochs <#epochs|15>                        # Number of epochs of main training"
  echo "                                                   # while reducing learning rate (determines #iterations, together"
  echo "                                                   # with --samples-per-iter and --num-jobs-nnet)"
  echo "  --num-epochs-extra <#epochs-extra|5>             # Number of extra epochs of training"
  echo "                                                   # after learning rate fully reduced"
  echo "  --initial-learning-rate <initial-learning-rate|0.02> # Learning rate at start of training, e.g. 0.02 for small"
  echo "                                                       # data, 0.01 for large data"
  echo "  --final-learning-rate  <final-learning-rate|0.004>   # Learning rate at end of training, e.g. 0.004 for small"
  echo "                                                   # data, 0.001 for large data"
  echo "  --num-hidden-layers <#hidden-layers|2>           # Number of hidden layers, e.g. 2 for 3 hours of data, 4 for 100hrs"
  echo "  --add-layers-period <#iters|2>                   # Number of iterations between adding hidden layers"
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
  echo "  --splice-width <width|4>                         # Number of frames on each side to append for feature input"
  echo "                                                   # (note: we splice processed, typically 40-dimensional frames"
  echo "  --lda-dim <dim|250>                              # Dimension to reduce spliced features to with LDA"
  echo "  --num-iters-final <#iters|20>                    # Number of final iterations to give to nnet-combine-fast to "
  echo "                                                   # interpolate parameters (the weights are learned with a validation set)"
  echo "  --first-component-power <power|1.0>              # Power applied to output of first p-norm layer... setting this to"
  echo "                                                   # 0.5 seems to help under some circumstances."
  echo "  --stage <stage|-9>                               # Used to run a partially-completed training process from somewhere in"
  echo "                                                   # the middle."

  
  exit 1;
fi

data=$1
lang=$2
alidir=$3
dir=$4

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

extra_opts=()
[ ! -z "$cmvn_opts" ] && extra_opts+=(--cmvn-opts "$cmvn_opts")
[ ! -z "$feat_type" ] && extra_opts+=(--feat-type $feat_type)
[ ! -z "$online_ivector_dir" ] && extra_opts+=(--online-ivector-dir $online_ivector_dir)
[ -z "$transform_dir" ] && transform_dir=$alidir
extra_opts+=(--transform-dir $transform_dir)
extra_opts+=(--splice-width $splice_width)

if [ $stage -le -4 ]; then
  echo "$0: calling get_lda.sh"
  steps/nnet2/get_lda.sh $lda_opts "${extra_opts[@]}" --cmd "$cmd" $data $lang $alidir $dir || exit 1;
fi

# these files will have been written by get_lda.sh
feat_dim=$(cat $dir/feat_dim) || exit 1;
ivector_dim=$(cat $dir/ivector_dim) || exit 1;
lda_dim=$(cat $dir/lda_dim) || exit 1;

if [ $stage -le -3 ] && [ -z "$egs_dir" ]; then
  echo "$0: calling get_egs.sh"
  steps/nnet2/get_egs.sh $egs_opts "${extra_opts[@]}" \
      --samples-per-iter $samples_per_iter \
      --num-jobs-nnet $num_jobs_nnet --stage $get_egs_stage \
      --cmd "$cmd" $egs_opts --io-opts "$io_opts" \
      $data $lang $alidir $dir || exit 1;
fi

if [ -z $egs_dir ]; then
  egs_dir=$dir/egs
fi

iters_per_epoch=`cat $egs_dir/iters_per_epoch`  || exit 1;
! [ $num_jobs_nnet -eq `cat $egs_dir/num_jobs_nnet` ] && \
  echo "$0: Warning: using --num-jobs-nnet=`cat $egs_dir/num_jobs_nnet` from $egs_dir"
num_jobs_nnet=`cat $egs_dir/num_jobs_nnet` || exit 1;


if ! [ $num_hidden_layers -ge 1 ]; then
  echo "Invalid num-hidden-layers $num_hidden_layers"
  exit 1
fi

if [ $stage -le -2 ]; then
  echo "$0: initializing neural net";
  lda_mat=$dir/lda.mat
  tot_input_dim=$[$feat_dim+$ivector_dim]

  online_preconditioning_opts="alpha=$alpha num-samples-history=$num_samples_history update-period=$update_period rank-in=$precondition_rank_in rank-out=$precondition_rank_out max-change-per-sample=$max_change_per_sample"

  stddev=`perl -e "print 1.0/sqrt($pnorm_input_dim);"`
  cat >$dir/nnet.config <<EOF
SpliceComponent input-dim=$tot_input_dim left-context=$splice_width right-context=$splice_width const-component-dim=$ivector_dim
FixedAffineComponent matrix=$lda_mat
AffineComponentPreconditionedOnline input-dim=$lda_dim output-dim=$pnorm_input_dim $online_preconditioning_opts learning-rate=$initial_learning_rate param-stddev=$stddev bias-stddev=$bias_stddev
PnormComponent input-dim=$pnorm_input_dim output-dim=$pnorm_output_dim p=$p
NormalizeComponent dim=$pnorm_output_dim
AffineComponentPreconditionedOnline input-dim=$pnorm_output_dim output-dim=$num_leaves $online_preconditioning_opts learning-rate=$initial_learning_rate param-stddev=0 bias-stddev=0
SoftmaxComponent dim=$num_leaves
EOF

  # to hidden.config it will write the part of the config corresponding to a
  # single hidden layer; we need this to add new layers. 
  cat >$dir/hidden.config <<EOF
AffineComponentPreconditionedOnline input-dim=$pnorm_output_dim output-dim=$pnorm_input_dim $online_preconditioning_opts learning-rate=$initial_learning_rate param-stddev=$stddev bias-stddev=$bias_stddev
PnormComponent input-dim=$pnorm_input_dim output-dim=$pnorm_output_dim p=$p
NormalizeComponent dim=$pnorm_output_dim
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

num_iters_reduce=$[$num_epochs * $iters_per_epoch];
num_iters_extra=$[$num_epochs_extra * $iters_per_epoch];
num_iters=$[$num_iters_reduce+$num_iters_extra]

echo "$0: Will train for $num_epochs + $num_epochs_extra epochs, equalling "
echo "$0: $num_iters_reduce + $num_iters_extra = $num_iters iterations, "
echo "$0: (while reducing learning rate) + (with constant learning rate)."


function set_target_objf_change {
  # nothing to do if $target_multiplier not set.
  [ "$target_multiplier" == "0" -o "$target_multiplier" == "0.0" ] && return;
  [ $x -le $finish_add_layers_iter ] && return;
  wait=2  # the compute_prob_{train,valid} from 2 iterations ago should
          # most likey be done even though we backgrounded them.
  [ $[$x-$wait] -le 0 ] && return;
  while true; do
    # Note: awk 'some-expression' is the same as: awk '{if(some-expression) print;}'
    train_prob=$(awk '(NF == 1)' < $dir/log/compute_prob_train.$[$x-$wait].log)
    valid_prob=$(awk '(NF == 1)' < $dir/log/compute_prob_valid.$[$x-$wait].log)
    if [ -z "$train_prob" ] || [ -z "$valid_prob" ]; then
      echo "$0: waiting until $dir/log/compute_prob_{train,valid}.$[$x-$wait].log are done"
      sleep 60
    else
      target_objf_change=$(perl -e '($train,$valid,$min_change,$multiplier)=@ARGV; if (!($train < 0.0) || !($valid < 0.0)) { print "0\n"; print STDERR "Error: invalid train or valid prob: $train_prob, $valid_prob\n"; exit(0); } else { print STDERR "train,valid=$train,$valid\n"; $proposed_target = $multiplier * ($train-$valid); if ($proposed_target < $min_change) { print "0"; } else { print $proposed_target; }}' -- "$train_prob" "$valid_prob" "$min_target_objf_change" "$target_multiplier")
      echo "On iter $x, (train,valid) probs from iter $[$x-$wait] were ($train_prob,$valid_prob), and setting target-objf-change to $target_objf_change."
      return;
    fi
  done
}

finish_add_layers_iter=$[$num_hidden_layers * $add_layers_period]
# This is when we decide to mix up from: halfway between when we've finished
# adding the hidden layers and the end of training.
mix_up_iter=$[($num_iters + $finish_add_layers_iter)/2]

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

x=0
target_objf_change=0 # relates to perturbed training.

while [ $x -lt $num_iters ]; do
  if [ $x -ge 0 ] && [ $stage -le $x ]; then
    # Set off jobs doing some diagnostics, in the background.
    $cmd $dir/log/compute_prob_valid.$x.log \
      nnet-compute-prob $dir/$x.mdl ark:$egs_dir/valid_diagnostic.egs &
    $cmd $dir/log/compute_prob_train.$x.log \
      nnet-compute-prob $dir/$x.mdl ark:$egs_dir/train_diagnostic.egs &
    if [ $x -gt 0 ] && [ ! -f $dir/log/mix_up.$[$x-1].log ]; then
      $cmd $dir/log/progress.$x.log \
        nnet-show-progress --use-gpu=no $dir/$[$x-1].mdl $dir/$x.mdl \
          ark:$egs_dir/train_diagnostic.egs '&&' \
        nnet-am-info $dir/$x.mdl &
    fi
    
    echo "Training neural net (pass $x)"

    if [ $x -gt 0 ] && \
      [ $x -le $[($num_hidden_layers-1)*$add_layers_period] ] && \
      [ $[($x-1) % $add_layers_period] -eq 0 ]; then
      mdl="nnet-init --srand=$x $dir/hidden.config - | nnet-insert $dir/$x.mdl - - |"
    else
      mdl=$dir/$x.mdl
    fi
    if [ $x -eq 0 ] || [ "$mdl" != "$dir/$x.mdl" ]; then
      # on iteration zero or when we just added a layer, use a smaller minibatch
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

    set_target_objf_change;  # only has effect if target_multiplier != 0
    if [ "$target_objf_change" != "0" ]; then
      [ ! -f $dir/within_covar.spmat ] && \
        echo "$0: expected $dir/within_covar.spmat to exist." && exit 1;
      perturb_suffix="-perturbed"
      perturb_opts="--target-objf-change=$target_objf_change --within-covar=$dir/within_covar.spmat"
    fi
    
    $cmd $parallel_opts JOB=1:$num_jobs_nnet $dir/log/train.$x.JOB.log \
      nnet-shuffle-egs --buffer-size=$shuffle_buffer_size --srand=$x \
      ark:$egs_dir/egs.JOB.$[$x%$iters_per_epoch].ark ark:- \| \
       nnet-train$parallel_suffix$perturb_suffix $parallel_train_opts $perturb_opts \
        --minibatch-size=$this_minibatch_size --srand=$x "$mdl" \
        ark:- $dir/$[$x+1].JOB.mdl \
      || exit 1;

    nnets_list=
    for n in `seq 1 $num_jobs_nnet`; do
      nnets_list="$nnets_list $dir/$[$x+1].$n.mdl"
    done

    learning_rate=`perl -e '($x,$n,$i,$f)=@ARGV; print ($x >= $n ? $f : $i*exp($x*log($f/$i)/$n));' $[$x+1] $num_iters_reduce $initial_learning_rate $final_learning_rate`;

    if $do_average; then
      $cmd $dir/log/average.$x.log \
        nnet-am-average $nnets_list - \| \
        nnet-am-copy --learning-rate=$learning_rate - $dir/$[$x+1].mdl || exit 1;
    else
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
       [ $[($x-1)%100] -ne 0  ] && [ $[$x-1] -le $[$num_iters-$num_iters_final] ]; then
      rm $dir/$[$x-1].mdl
    fi
  fi
  x=$[$x+1]
done

# Now do combination.
# At the end, final.mdl will be a combination of the last e.g. 10 models.
nnets_list=()
if [ $num_iters_final -gt $num_iters_extra ]; then
  echo "Setting num_iters_final=$num_iters_extra"
fi
start=$[$num_iters-$num_iters_final+1]
for x in `seq $start $num_iters`; do
  idx=$[$x-$start]
  if [ $x -gt $mix_up_iter ]; then
    nnets_list[$idx]=$dir/$x.mdl # "nnet-am-copy --remove-dropout=true $dir/$x.mdl - |"
  fi
done

if [ $stage -le $num_iters ]; then
  echo "Doing final combination to produce final.mdl"
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

if $cleanup; then
  echo Cleaning up data
  if [ $egs_dir == "$dir/egs" ]; then
    steps/nnet2/remove_egs.sh $dir/egs
  fi
fi
