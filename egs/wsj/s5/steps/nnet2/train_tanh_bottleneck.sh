#!/bin/bash

# Copyright 2012  Johns Hopkins University (Author: Daniel Povey).  Apache 2.0.
#	    2014  Pegah Ghahremani
# This script trains a fairly vanilla network with tanh nonlinearities to generate bottleneck features

# Begin configuration section.
cmd=run.pl
num_epochs=15    # Number of epochs during which we reduce
                   # the learning rate; number of iteration is worked out from this.
num_epochs_extra=5 # Number of epochs after we stop reducing
                   # the learning rate.
num_iters_final=20 # Maximum number of final iterations to give to the
                   # optimization over the validation set.
initial_learning_rate=0.04
final_learning_rate=0.004
bias_stddev=0.5
shrink_interval=5 # shrink every $shrink_interval iters except while we are
                  # still adding layers, when we do it every iter.
shrink=true
num_frames_shrink=2000 # note: must be <= --num-frames-diagnostic option to get_egs.sh, if
                       # given.
final_learning_rate_factor=0.5 # Train the two last layers of parameters half as
                               # fast as the other layers.

hidden_layer_dim=1024 #  You may want this larger, e.g. 1024 or 2048.

bottleneck_dim=42  # bottleneck layer dimension
minibatch_size=128 # by default use a smallish minibatch size for neural net
                   # training; this controls instability which would otherwise
                   # be a problem with multi-threaded update.  Note: it also
                   # interacts with the "preconditioned" update which generally
                   # works better with larger minibatch size, so it's not
                   # completely cost free.

samples_per_iter=200000 # each iteration of training, see this many samples
                        # per job.  This option is passed to get_egs.sh
num_jobs_nnet=16   # Number of neural net jobs to run in parallel.  This option
                   # is passed to get_egs.sh.
get_egs_stage=0

shuffle_buffer_size=5000 # This "buffer_size" variable controls randomization of the samples
                # on each iter.  You could set it to 0 or to a large value for complete
                # randomization, but this would both consume memory and cause spikes in
                # disk I/O.  Smaller is easier on disk and memory but less random.  It's
                # not a huge deal though, as samples are anyway randomized right at the start.

add_layers_period=2 # by default, add new layers every 2 iterations.
num_hidden_layers=3
bottleneck_layer_num=$num_hidden_layers-2 # bottleneck layer number between hidden layer
                                        # eg. 1024|1024|42|1024 bottleneck_layer_num = 2

modify_learning_rates=false
last_layer_factor=0.1 # relates to modify_learning_rates.
first_layer_factor=1.0 # relates to modify_learning_rates.
stage=-5

io_opts="--max-jobs-run 5" # for jobs with a lot of I/O, limits the number running at one time.   These don't
splice_width=4 # meaning +- 4 frames on each side for second LDA
randprune=4.0 # speeds up LDA.
alpha=4.0
max_change=10.0
mix_up=0 # Number of components to mix up to (should be > #tree leaves, if
        # specified.)
num_threads=16
parallel_opts="--num-threads 16 --mem 1G" # by default we use 16 threads; this lets the queue know.
  # note: parallel_opts doesn't automatically get adjusted if you adjust num-threads.
combine_opts="--mem 12G"
cleanup=true
egs_dir=
lda_opts=
egs_opts=
transform_dir=
nj=
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
  echo "  --initial-num-hidden-layers <#hidden-layers|1>   # Number of hidden layers to start with."
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
  echo "  --parallel-opts <opts|\"--num-threads 16 --mem 1G\">      # extra options to pass to e.g. queue.pl for processes that"
  echo "                                                   # use multiple threads... "
  echo "  --io-opts <opts|\"--max-jobs-run 10\">                      # Options given to e.g. queue.pl for jobs that do a lot of I/O."
  echo "  --minibatch-size <minibatch-size|128>            # Size of minibatch to process (note: product with --num-threads"
  echo "                                                   # should not get too large, e.g. >2k)."
  echo "  --samples-per-iter <#samples|200000>             # Number of samples of data to process per iteration, per"
  echo "                                                   # process."
  echo "  --splice-width <width|4>                         # Number of frames on each side to append for feature input"
  echo "                                                   # (note: we splice processed, typically 40-dimensional frames"
  echo "  --lda-dim <dim|250>                              # Dimension to nsformreduce spliced features to with LDA"
  echo "  --num-iters-final <#iters|10>                    # Number of final iterations to give to nnet-combine-fast to "
  echo "                                                   # interpolate parameters (the weights are learned with a validation set)"
  echo "  --num-utts-subset <#utts|300>                    # Number of utterances in subsets used for validation and diagnostics"
  echo "                                                   # (the validation subset is held out from training)"
  echo "  --num-frames-diagnostic <#frames|4000>           # Number of frames used in computing (train,valid) diagnostics"
  echo "  --num-valid-frames-combine <#frames|10000>       # Number of frames used in getting combination weights at the"
  echo "                                                   # very end."
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
num_leaves=`am-info $alidir/final.mdl 2>/dev/null | awk '/number of pdfs/{print $NF}'` || exit 1;

nj=`cat $alidir/num_jobs` || exit 1;  # number of jobs in alignment dir...
# in this dir we'll have just one job.
sdata=$data/split$nj
utils/split_data.sh $data $nj

mkdir -p $dir/log
splice_opts=`cat $alidir/splice_opts 2>/dev/null`
cp $alidir/final.mat $dir 2>/dev/null
cp $alidir/splice_opts $dir 2>/dev/null
cmvn_opts=`cat $alidir/cmvn_opts 2>/dev/null`
cp $alidir/cmvn_opts $dir 2>/dev/null
cp $alidir/tree $dir

utils/lang/check_phones_compatible.sh $lang/phones.txt $alidir/phones.txt || exit 1;
cp $lang/phones.txt $dir || exit 1;

truncate_comp_num=$[2*$num_hidden_layers+1]
if [ $stage -le -4 ]; then
  echo "$0: calling get_lda.sh"
  steps/nnet2/get_lda.sh $lda_opts --splice-width $splice_width --cmd "$cmd" $data $lang $alidir $dir || exit 1;
fi

# these files will have been written by get_lda.sh
feat_dim=`cat $dir/feat_dim` || exit 1;
lda_dim=`cat $dir/lda_dim` || exit 1;

if [ $stage -le -3 ] && [ -z "$egs_dir" ]; then
  echo "$0: calling get_egs.sh"
  [ ! -z $transform_dir ] && $transform_dir_opt="--transform-dir $transform_dir";
  steps/nnet2/get_egs.sh $transform_dir_opt --samples-per-iter $samples_per_iter \
      --num-jobs-nnet $num_jobs_nnet --splice-width $splice_width --stage $get_egs_stage \
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

  stddev=`perl -e "print 1.0/sqrt($hidden_layer_dim);"`
  cat >$dir/nnet.config <<EOF
SpliceComponent input-dim=$feat_dim left-context=$splice_width right-context=$splice_width const-component-dim=0
FixedAffineComponent matrix=$lda_mat
AffineComponentPreconditioned input-dim=$lda_dim output-dim=$hidden_layer_dim alpha=$alpha max-change=$max_change learning-rate=$initial_learning_rate param-stddev=$stddev bias-stddev=$bias_stddev
TanhComponent dim=$hidden_layer_dim
AffineComponentPreconditioned input-dim=$hidden_layer_dim output-dim=$num_leaves alpha=$alpha max-change=$max_change learning-rate=$initial_learning_rate param-stddev=0 bias-stddev=0
SoftmaxComponent dim=$num_leaves
EOF

  # to hidden.config it will write the part of the config corresponding to a
  # single hidden layer; we need this to add new layers.
  cat >$dir/hidden.config <<EOF
AffineComponentPreconditioned input-dim=$hidden_layer_dim output-dim=$hidden_layer_dim alpha=$alpha max-change=$max_change learning-rate=$initial_learning_rate param-stddev=$stddev bias-stddev=$bias_stddev
TanhComponent dim=$hidden_layer_dim
EOF
  bottleneck_stddev=`perl -e "print 1.0/sqrt($bottleneck_dim);"`
  # bnf.config it will write the part of th config corresponding to a
  # bottleneck layer; we need this to add bottleneck layer.
  cat >$dir/bnf.config <<EOF
AffineComponentPreconditioned input-dim=$hidden_layer_dim output-dim=$bottleneck_dim alpha=$alpha max-change=$max_change learning-rate=$initial_learning_rate param-stddev=$stddev bias-stddev=$bias_stddev
AffineComponentPreconditioned input-dim=$bottleneck_dim output-dim=$hidden_layer_dim alpha=$alpha max-change=$max_change learning-rate=$initial_learning_rate param-stddev=$bottleneck_stddev bias-stddev=$bias_stddev
TanhComponent dim=$hidden_layer_dim
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
echo num_iters = $num_iters
echo "$0: Will train for $num_epochs + $num_epochs_extra epochs, equalling "
echo "$0: $num_iters_reduce + $num_iters_extra = $num_iters iterations, "
echo "$0: (while reducing learning rate) + (with constant learning rate)."

# This is when we decide to mix up from: halfway between when we've finished
# adding the hidden layers and the end of training.
finish_add_layers_iter=$[($num_hidden_layers-$initial_num_hidden_layers+1)*$add_layers_period]
first_modify_iter=$[$finish_add_layers_iter + $add_layers_period]
mix_up_iter=$[($num_iters + $finish_add_layers_iter)/2]
truncate_comp_num=$[2*$num_hidden_layers+1]
if [ $num_threads -eq 1 ]; then
  train_suffix="-simple" # this enables us to use GPU code if
                         # we have just one thread.
else
  train_suffix="-parallel --num-threads=$num_threads"
fi

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
    if [ $x -gt 0 ] && \
      [ $x -le $[($num_hidden_layers-1)*$add_layers_period] ] && \
      [ $[($x-1) % $add_layers_period] -eq 0 ]; then
      if [ $[($x-1) / $add_layers_period] -eq $[($num_hidden_layers-2)] ]; then
        echo bnf layer with x = $x
        mdl="nnet-init --srand=$x $dir/bnf.config - | nnet-insert $dir/$x.mdl - - |"
      else
        mdl="nnet-init --srand=$x $dir/hidden.config - | nnet-insert $dir/$x.mdl - - |"
      fi
    else
      mdl=$dir/$x.mdl
    fi

    $cmd $parallel_opts JOB=1:$num_jobs_nnet $dir/log/train.$x.JOB.log \
      nnet-shuffle-egs --buffer-size=$shuffle_buffer_size --srand=$x \
      ark:$egs_dir/egs.JOB.$[$x%$iters_per_epoch].ark ark:- \| \
      nnet-train$train_suffix \
         --minibatch-size=$minibatch_size --srand=$x "$mdl" \
        ark:- $dir/$[$x+1].JOB.mdl \
      || exit 1;

    nnets_list=
    for n in `seq 1 $num_jobs_nnet`; do
      nnets_list="$nnets_list $dir/$[$x+1].$n.mdl"
    done

    learning_rate=`perl -e '($x,$n,$i,$f)=@ARGV; print ($x >= $n ? $f : $i*exp($x*log($f/$i)/$n));' $[$x+1] $num_iters_reduce $initial_learning_rate $final_learning_rate`;
    last_layer_learning_rate=`perl -e "print $learning_rate * $final_learning_rate_factor;"`;
    nnet-am-info $dir/$[$x+1].1.mdl > $dir/foo  2>/dev/null || exit 1
    nu=`cat $dir/foo | grep num-updatable-components | awk '{print $2}'`
    na=`cat $dir/foo | grep -v Fixed | grep AffineComponent | wc -l`
    # na is number of last updatable AffineComponent layer [one-based, counting only
    # updatable components.]
    # The last two layers will get this (usually lower) learning rate.
    lr_string="$learning_rate"
    for n in `seq 2 $nu`; do
      if [ $n -eq $na ] || [ $n -eq $[$na-1] ]; then lr=$last_layer_learning_rate;
      else lr=$learning_rate; fi
      lr_string="$lr_string:$lr"
    done

    $cmd $dir/log/average.$x.log \
      nnet-am-average $nnets_list - \| \
      nnet-am-copy --learning-rates=$lr_string - $dir/$[$x+1].mdl || exit 1;

    if $modify_learning_rates && [ $x -ge $first_modify_iter ]; then
      $cmd $dir/log/modify_learning_rates.$x.log \
        nnet-modify-learning-rates --last-layer-factor=$last_layer_factor \
          --first-layer-factor=$first_layer_factor --average-learning-rate=$learning_rate \
        $dir/$x.mdl $dir/$[$x+1].mdl $dir/$[$x+1].mdl || exit 1;
    fi

    if $shrink && [ $[$x % $shrink_interval] -eq 0 ]; then
      mb=$[($num_frames_shrink+$num_threads-1)/$num_threads]
      $cmd $parallel_opts $dir/log/shrink.$x.log \
        nnet-subset-egs --n=$num_frames_shrink --randomize-order=true --srand=$x \
          ark:$egs_dir/train_diagnostic.egs ark:-  \| \
        nnet-combine-fast --use-gpu=no --num-threads=$num_threads --verbose=3 --minibatch-size=$mb \
          $dir/$[$x+1].mdl ark:- $dir/$[$x+1].mdl || exit 1;
    else
      # On other iters, do nnet-am-fix which is much faster and has roughly
      # the same effect.
      nnet-am-fix $dir/$[$x+1].mdl $dir/$[$x+1].mdl 2>$dir/log/fix.$x.log
    fi

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
  # Below, use --use-gpu=no to disable nnet-combine-fast from using a GPU, as
  # if there are many models it can give out-of-memory error; set num-threads to 8
  # to speed it up (this isn't ideal...)
  this_num_threads=$num_threads
  [ $this_num_threads -lt 8 ] && this_num_threads=8
  num_egs=`nnet-copy-egs ark:$egs_dir/combine.egs ark:/dev/null 2>&1 | tail -n 1 | awk '{print $NF}'`
  mb=$[($num_egs+$this_num_threads-1)/$this_num_threads]
  [ $mb -gt 512 ] && mb=512
  $cmd $parallel_opts $combine_opts $dir/log/combine.log \
    nnet-combine-fast --use-gpu=no --num-threads=$this_num_threads \
      --verbose=3 --minibatch-size=$mb "${nnets_list[@]}" ark:$egs_dir/combine.egs \
      $dir/final.mdl || exit 1;
fi

# Compute the probability of the final, combined model with
# the same subset we used for the previous compute_probs, as the
# different subsets will lead to different probs.
$cmd $dir/log/compute_prob_valid.final.log \
  nnet-compute-prob $dir/final.mdl ark:$egs_dir/valid_diagnostic.egs &
$cmd $dir/log/compute_prob_train.final.log \
  nnet-compute-prob $dir/final.mdl ark:$egs_dir/train_diagnostic.egs &

sleep 2

echo Done

if $cleanup; then
  echo Cleaning up data
  if [ $egs_dir == "$dir/egs" ]; then
    steps/nnet2/remove_egs.sh $dir/egs
  fi
  echo Removing most of the models
  for x in `seq 0 $num_iters`; do
    if [ $[$x%10] -ne 0 ] && [ $x -lt $[$num_iters-$num_iters_final+1] ]; then
       # delete all but every 10th model; don't delete the ones which combine to form the final model.
      rm $dir/$x.mdl
    fi
  done
fi

name=`basename $data`
if [ -f $dir/final.mdl ]; then
  nnet-to-raw-nnet --truncate=$truncate_comp_num $dir/final.mdl $dir/final.raw
else
  echo "$0: we require final.mdl in source dir $dir"
fi

