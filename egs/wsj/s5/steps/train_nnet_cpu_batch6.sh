#!/bin/bash

# Copyright 2012  Johns Hopkins University (Author: Daniel Povey).  Apache 2.0.

# *batch6.sh is as *batch4.sh, but using nnet-am-fix on each batcvh iteration,
# to prevent sigmoids from "maxing out" (this was more of an issue in the
# Switchboard setup.).

# *batch4.sh is as *batch2.sh, but with a couple of changes: some changes
# for speed- different ways of getting validation subsets and doing the "combine"
# thing (nnet-combine-fast), but also using 2 different subsets: a training-data
# subset and a validation-data subset.

# *batch2.sh is as *batch.sh but every iteration, we only take a fraction
# of the data (by default 1/2, but changeable with  --data-parts

# this was modified from train_nnet_cpu_parallel10d.sh
# We start off with SGD training, but after a few iterations of that we switch
# to a batch method.  On each iteration it computes the gradient, and also a
# preconditioner (this consists of a matrix transform at the input and output side
# of each weight matrix).  The method is kind of like a more general version of L-BFGS;
# it uses a validation set to optimize over a subspace formed by the last N models,
# the last N gradients (each times the preconditioner, which is like an inverse
# approximate Hessian), and the last N (model times preconditioners)-- we throw these
# in because they are terms that would arise in "modified gradient" terms containing
# l2 regularization.


# Begin configuration section.
cmd=run.pl
num_sgd_iters=5 # Number of SGD iterations
num_batch_iters=30 # Number of iterations of the batch update.
n=10 # This corresponds roughtly to the value "N" in L-BFGS and the like; it's
     # a number of iterations we keep in the subspace we optimize over on each iter.
data_parts=2

initial_learning_rate=0.02 # Initial learning rate of SGD phase.
final_learning_rate=0.005 # Final learning rate of SGD phase-- still quite high.

num_valid_utts=300    # held-out utterances, used only for diagnostics.
num_valid_frames=2000 # a subset of the frames in "valid_utts", used only
                      # for estimating shrinkage parameters for the
                      # SGD phase and the batch phase.
num_train_frames=4000 # for combination weights in batch phase.

num_precon_frames=20000 # A subset of frames for computing the preconditioner.
precon_alpha=0.1 # alpha value for nnet-precondition
fix_opts=
apply_shrinking=true

minibatch_size=1024
minibatches_per_phase=100 # only affects diagnostic messages during SGD phase.
samples_per_iteration=200000 # During SGD phase of training, see this many samples
                             # per job.
num_jobs_sgd=8 # Number of SGD jobs to run in parallel (we average the parameters
               # after that.)
    # Note: for the batch phase of training, the number of jobs is determined
    # by the alignment directory.

add_layers_period=2 # by default, add new layers every 2 iterations.
num_hidden_layers=2
initial_num_hidden_layers=1  # we'll add the rest one by one.
num_parameters=2000000 # 2 million parameters by default.  You'll probably
                       # want to provide this parameter.
stage=-5
realign_iters=""
beam=10  # for realignment.
retry_beam=40
scale_opts="--transition-scale=1.0 --acoustic-scale=0.1 --self-loop-scale=0.1"
parallel_opts=
num_threads=8  # Number of threads for each process of nnet-gradient and nnet-combine to use.
nnet_config_opts=  # Options to pass to utils/nnet-cpu/make_nnet_config_preconditioned.pl
splice_width=4 # meaning +- 4 frames on each side for second LDA
lda_dim=250
randprune=4.0 # speeds up LDA.
# If you specify alpha, then we'll do the "preconditioned" update.
alpha=
shrink=true
mix_up=0 # Number of components to mix up to (should be > #tree leaves, if
        # specified.)
# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;


if [ $# != 4 ]; then
  echo "Usage: steps/train_nnet_cpu_batch4.sh <data> <lang> <ali-dir> <exp-dir>"
  echo " e.g.: steps/train_nnet_cpu_batch4.sh data/train_si84 data/lang \\"
  echo "                      exp/tri3b_ali_si84 exp/ubm4a/final.ubm exp/sgmm4a"
  echo "main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config containing options"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --num-iters <#iters>                             # Number of iterations of training"
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
oov=`cat $lang/oov.int`
feat_dim=`gmm-info $alidir/final.mdl 2>/dev/null | awk '/feature dimension/{print $NF}'` || exit 1;
num_leaves=`gmm-info $alidir/final.mdl 2>/dev/null | awk '/number of pdfs/{print $NF}'` || exit 1;
silphonelist=`cat $lang/phones/silence.csl` || exit 1;

nj=`cat $alidir/num_jobs` || exit 1;  # number of jobs in alignment dir...
# We'll use this number of jobs for the batch part of training.
sdata=$data/split$nj

mkdir -p $dir/log
echo $nj > $dir/num_jobs
splice_opts=`cat $alidir/splice_opts 2>/dev/null`
cp $alidir/splice_opts $dir 2>/dev/null
cp $alidir/final.mat $dir 2>/dev/null # any LDA matrix...
cp $alidir/tree $dir


# Get list of validation utterances. 
awk '{print $1}' $data/utt2spk | utils/shuffle_list.pl | head -$num_valid_utts \
    > $dir/valid_uttlist || exit 1;

## Set up features.  Note: these are different from the normal features
## because we have one rspecifier that has the features for the entire
## training set, not separate ones for each batch.
if [ -f $alidir/final.mat ]; then feat_type=lda; else feat_type=delta; fi
echo "$0: feature type is $feat_type"

case $feat_type in
  delta) feats="ark,s,cs:utils/filter_scp.pl --exclude $dir/valid_uttlist $data/feats.scp | apply-cmvn --norm-vars=false --utt2spk=ark:$data/utt2spk scp:$data/cmvn.scp scp:- ark:- | add-deltas ark:- ark:- |"
     split_feats="ark,s,cs:utils/filter_scp.pl --exclude $dir/valid_uttlist $sdata/JOB/feats.scp | apply-cmvn --norm-vars=false --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:- ark:- | add-deltas ark:- ark:- |"
    valid_feats="ark,s,cs:utils/filter_scp.pl $dir/valid_uttlist $data/feats.scp | apply-cmvn --norm-vars=false --utt2spk=ark:$data/utt2spk scp:$data/cmvn.scp scp:- ark:- | add-deltas ark:- ark:- |"
   ;;
  lda) feats="ark,s,cs:utils/filter_scp.pl --exclude $dir/valid_uttlist $data/feats.scp | apply-cmvn --norm-vars=false --utt2spk=ark:$data/utt2spk scp:$data/cmvn.scp scp:- ark:- | splice-feats $splice_opts ark:- ark:- | transform-feats $dir/final.mat ark:- ark:- |"
      split_feats="ark,s,cs:utils/filter_scp.pl --exclude $dir/valid_uttlist $sdata/JOB/feats.scp | apply-cmvn --norm-vars=false --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:- ark:- | splice-feats $splice_opts ark:- ark:- | transform-feats $dir/final.mat ark:- ark:- |"
      valid_feats="ark,s,cs:utils/filter_scp.pl $dir/valid_uttlist $data/feats.scp | apply-cmvn --norm-vars=false --utt2spk=ark:$data/utt2spk scp:$data/cmvn.scp scp:- ark:- | splice-feats $splice_opts ark:- ark:- | transform-feats $dir/final.mat ark:- ark:- |"
    cp $alidir/final.mat $dir    
    ;;
  *) echo "$0: invalid feature type $feat_type" && exit 1;
esac
if [ -f $alidir/trans.1 ]; then
  echo "$0: using transforms from $alidir"
  feats="$feats transform-feats --utt2spk=ark:$data/utt2spk 'ark:cat $alidir/trans.*|' ark:- ark:- |"
  split_feats="$split_feats transform-feats --utt2spk=ark:$sdata/JOB/utt2spk ark:$alidir/trans.JOB ark:- ark:- |"
  valid_feats="$valid_feats transform-feats --utt2spk=ark:$data/utt2spk 'ark:cat $alidir/trans.*|' ark:- ark:- |"
fi

## Do LDA on top of whatever features we already have; store the matrix which
## we'll put into the neural network as a constant.

if [ $stage -le -5 ]; then
  echo "Accumulating LDA statistics."
  $cmd JOB=1:$nj $dir/log/lda_acc.JOB.log \
    ali-to-post "ark:gunzip -c $alidir/ali.JOB.gz|" ark:- \| \
      weight-silence-post 0.0 $silphonelist $alidir/final.mdl ark:- ark:- \| \
      acc-lda --rand-prune=$randprune $alidir/final.mdl "$split_feats splice-feats --left-context=$splice_width --right-context=$splice_width ark:- ark:- |" ark,s,cs:- \
       $dir/lda.JOB.acc || exit 1;
  est-lda --dim=$lda_dim $dir/lda.mat $dir/lda.*.acc \
      2>$dir/log/lda_est.log || exit 1;
  rm $dir/lda.*.acc
fi


##
if [ $initial_num_hidden_layers -gt $num_hidden_layers ]; then
  echo "Initial num-hidden-layers $initial_num_hidden_layers is greater than final number $num_hidden_layers";
  exit 1;
fi

if [ $stage -le -4 ]; then
  echo "$0: initializing neural net";
  # to hidden.config it will write the part of the config corresponding to a
  # single hidden layer; we need this to add new layers.
  if [ ! -z "$alpha" ]; then
    utils/nnet-cpu/make_nnet_config_preconditioned.pl --alpha $alpha $nnet_config_opts \
      --learning-rate $initial_learning_rate \
      --lda-mat $splice_width $lda_dim $dir/lda.mat \
      --initial-num-hidden-layers $initial_num_hidden_layers $dir/hidden_layer.config \
      $feat_dim $num_leaves $num_hidden_layers $num_parameters \
      > $dir/nnet.config || exit 1;
  else
    utils/nnet-cpu/make_nnet_config.pl $nnet_config_opts \
      --learning-rate $initial_learning_rate \
      --lda-mat $splice_width $lda_dim $dir/lda.mat \
      --initial-num-hidden-layers $initial_num_hidden_layers $dir/hidden_layer.config \
      $feat_dim $num_leaves $num_hidden_layers $num_parameters \
      > $dir/nnet.config || exit 1;
  fi
  $cmd $dir/log/nnet_init.log \
     nnet-am-init $alidir/tree $lang/topo "nnet-init $dir/nnet.config -|" \
       $dir/0.mdl || exit 1;
fi

if [ $stage -le -3 ]; then
  echo "Training transition probabilities and setting priors"
  $cmd $dir/log/train_trans.log \
    nnet-train-transitions $dir/0.mdl "ark:gunzip -c $alidir/ali.*.gz|" $dir/0.mdl \
    || exit 1;
fi

if [ $stage -le -2 ]; then
  echo "Compiling graphs of transcripts"
  $cmd JOB=1:$nj $dir/log/compile_graphs.JOB.log \
    compile-train-graphs $dir/tree $dir/0.mdl  $lang/L.fst  \
     "ark:utils/sym2int.pl --map-oov $oov -f 2- $lang/words.txt < $data/split$nj/JOB/text |" \
      "ark:|gzip -c >$dir/fsts.JOB.gz" || exit 1;
fi

cp $alidir/ali.*.gz $dir


nnet_context_opts="--left-context=`nnet-am-info $dir/0.mdl 2>/dev/null | grep -w left-context | awk '{print $2}'` --right-context=`nnet-am-info $dir/0.mdl 2>/dev/null | grep -w right-context | awk '{print $2}'`" || exit 1;

if [ $stage -le -1 ]; then
  echo "Creating subset of frames of validation set (in background)"
  $cmd $dir/log/create_valid_subset.log \
    nnet-get-egs $nnet_context_opts  \
      "$valid_feats" "ark,cs:gunzip -c $dir/ali.*.gz | ali-to-pdf $dir/0.mdl ark:- ark:- | ali-to-post ark:- ark:- |" \
     ark:- \| \
    nnet-subset-egs --srand=0 --n=$num_valid_frames ark:- ark:$dir/valid.egs &

  echo "Creating subset of frames of training set (in background)"
  $cmd $dir/log/create_train_subset.log \
    nnet-get-egs $nnet_context_opts  \
      "$feats" "ark,cs:gunzip -c $dir/ali.*.gz | ali-to-pdf $dir/0.mdl ark:- ark:- | ali-to-post ark:- ark:- |" \
     ark:- \| \
    nnet-subset-egs --srand=0 --n=$num_train_frames ark:- ark:$dir/train_subset.egs &

  echo "Creating subset of frames of training set, for computing preconditioners (in background)"
  $cmd $dir/log/create_precon_subset.log \
    nnet-get-egs $nnet_context_opts \
       "$feats" "ark,cs:gunzip -c $dir/ali.*.gz | ali-to-pdf $dir/0.mdl ark:- ark:- | ali-to-post ark:- ark:- |" \
     ark:- \| \
    nnet-subset-egs --srand=0 --n=$num_precon_frames ark:- ark:$dir/precon.egs &
fi

mix_up_iter=$[($num_hidden_layers-$initial_num_hidden_layers+1)*$add_layers_period + 1]
if [ $mix_up_iter -ge $num_sgd_iters ]; then
  echo "--num-sgd-iters is too small $num_sgd_iters, for these settings need more than $mix_up_iter"
fi

x=-1 # iterations start from 0 but
     # we always start off the "randomization" on the previous iteration.
while [ $x -lt $num_sgd_iters ]; do
  echo "Pass $x (SGD training)  "
  # note: archive for aligments won't be sorted as the shell glob "*" expands
  # them in alphabetic not numeric order, so we can't use ark,s,cs: below, only
  # ark,cs which means the features are in sorted order [hence alignments will
  # be called in sorted order (cs).

  wait $last_randomize_process # wait for any randomization from last time.
  y=$[$x+1];
  if [ $stage -le $y ] && [ $y -lt $num_sgd_iters ]; then # start off randomization for next time.
    egs_list=
    for n in `seq 1 $num_jobs_sgd`; do
       egs_list="$egs_list ark:$dir/egs.$y.tmp.$n"
    done
    # run the next randomization in the background, so it can run
    # while we're doing the training and combining.
    $cmd $parallel_opts $dir/log/randomize.$y.log \
     nnet-randomize-frames \
        $nnet_context_opts --num-samples=$[$samples_per_iteration*$num_jobs_sgd] \
      --srand=$y "$feats" \
       "ark,cs:gunzip -c $dir/ali.*.gz | ali-to-pdf $alidir/final.mdl ark:- ark:- | ali-to-post ark:- ark:- |" ark:- \| \
       nnet-copy-egs ark:- $egs_list &
    last_randomize_process=$! # process-id of the process what we just spawned.
  fi


  if [ $x -ge 0 ] && [ $stage -le $x ]; then
    # Set off a job that does diagnostics, in the background.
    $cmd $parallel_opts $dir/log/compute_prob.$x.log \
      nnet-compute-prob $dir/$x.mdl ark:$dir/valid.egs &

    echo "Training neural net (pass $x)"
    if [ $x -gt 0 ] && \
       [ $x -le $[($num_hidden_layers-$initial_num_hidden_layers)*$add_layers_period] ] && \
       [ $[($x-1) % $add_layers_period] -eq 0 ]; then
      mdl="nnet-init --srand=$x $dir/hidden_layer.config - | nnet-insert $dir/$x.mdl - - |"
    else
      mdl=$dir/$x.mdl
    fi
    m=$minibatches_per_phase

    $cmd $parallel_opts JOB=1:$num_jobs_sgd $dir/log/train.$x.JOB.log \
      nnet-train-simple \
        --minibatch-size=$minibatch_size --minibatches-per-phase=$m \
        --verbose=2 "$mdl" ark:$dir/egs.$x.tmp.JOB $dir/$[$x+1].JOB.mdl \
       || exit 1;

    egs_list=
    nnets_list=
    for n in `seq 1 $num_jobs_sgd`; do
      egs_list="$egs_list $dir/egs.$x.tmp.$n"
      nnets_list="$nnets_list $dir/$[$x+1].$n.mdl"
    done

    learning_rate=`perl -e '($x,$n,$i,$f)=@ARGV; print $i*exp($x*log($f/$i)/$n);' $[$x+1] $num_sgd_iters $initial_learning_rate $final_learning_rate`;

    $cmd $parallel_opts $dir/log/average.$x.log \
       nnet-am-average $nnets_list - \| \
       nnet-am-copy --learning-rate=$learning_rate - $dir/$[$x+1].mdl || exit 1;

    # nnet-combine-fast is the same as nnet-shrink when applied to
    # one model, but faster.
    if $shrink; then
      $cmd $parallel_opts $dir/log/shrink.$x.log \
        nnet-combine-fast --num-threads=$num_threads $dir/$[$x+1].mdl ark:$dir/valid.egs $dir/$[$x+1].mdl || exit 1;
    fi
    if [ "$mix_up" -gt 0 ] && [ $x -eq $mix_up_iter ]; then
      # mix up.
      echo Mixing up from $num_leaves to $mix_up components
      $cmd $dir/log/mix_up.$x.log \
        nnet-am-mixup --min-count=10 --num-mixtures=$mix_up \
         $dir/$[$x+1].mdl $dir/$[$x+1].mdl || exit 1;
    fi
    rm $nnets_list $egs_list
  fi
  x=$[$x+1]
done

num_tot_iters=$[$num_sgd_iters + $num_batch_iters];
while [ $x -lt $num_tot_iters ]; do
  echo "Pass $x (Batch training)"
  if [ $stage -le $x ]; then
    # Realign, if requested on this iteration.
    if echo $realign_iters | grep -w $x >/dev/null; then
      wait; # Wait for any "randomize" processes that are running, as
            # these could crash if we change the alignments on disk.   
      echo "Realigning data (pass $x)"
      $cmd JOB=1:$nj $dir/log/align.$x.JOB.log \
        nnet-align-compiled $scale_opts --beam=$beam --retry-beam=$retry_beam "$dir/$x.mdl" \
         "ark:gunzip -c $dir/fsts.JOB.gz|" "$split_feats" \
        "ark:|gzip -c >$dir/ali.JOB.gz" || exit 1;
    fi

    wait; # In case we were waiting on the job creating
          # precon.egs
    [ ! -s $dir/precon.egs ] && \
       echo "File $dir/precon.egs is empty or non-existent" && exit 1;
   
    # Get the preconditioner in the background.  Note: instead of writing directly
    # to $dir/$x.precon, we pipe it into a command using nnet-precondition
    # that has the effect of pre-computing some stuff that will help
    # the preconditioner be applied faster.
    $cmd $parallel_opts $dir/log/precon.$x.log \
       nnet-get-preconditioner --minibatch-size=$minibatch_size \
         $dir/$x.mdl ark:$dir/precon.egs "|nnet-precondition --alpha=$precon_alpha --renormalize=false - $dir/$x.mdl /dev/null $dir/$x.precon" &

    # Now accumulate gradient stats.
    $cmd $parallel_opts JOB=1:$nj $dir/log/gradient.$x.JOB.log \
       nnet-get-egs $nnet_context_opts "$split_feats" \
         "ark:gunzip -c $dir/ali.JOB.gz | ali-to-pdf $alidir/final.mdl ark:- ark:- | ali-to-post ark:- ark:- |" ark:- \| \
       nnet-select-egs --k=$[$x%$data_parts] --n=$data_parts ark:- ark:- \| \
       nnet-gradient --minibatch-size=$minibatch_size --num-threads=$num_threads \
         $dir/$x.mdl ark:- $dir/$x.JOB.gradient || exit 1;
    all_gradients=
    for y in `seq 1 $nj`; do all_gradients="$all_gradients $dir/$x.$y.gradient"; done
    $cmd $dir/log/sum.$x.log \
      nnet-am-average --sum=true $all_gradients $dir/$x.gradient || exit 1;
    rm $dir/$x.*.gradient # Remove partial gradients.

    wait; # For the job getting $dir/$x.precon
    [ ! -s $dir/$x.precon ] && \
       echo "File $dir/$x.precon is empty or non-existent" && exit 1;


    preconditioner=$dir/$x.precon
    # all models is a set of models and gradients that we'll give
    # to the
    all_models=() # empty array
    # Do these in backwards order and the first model (index 0)
    # will be specified as the point to start the optimization at.
    for y in `seq $x -1 $[$x - $n]`; do
      if [ $y -ge $num_sgd_iters ]; then
         all_models+=($dir/$y.mdl) # the model (append to array)
         # Note: we only give the "alpha" argument for didactic reasons.  It won't
         # have an effect, what matters is the alpha value we used above.
         all_models+=("nnet-precondition --alpha=$precon_alpha --normalize=true $preconditioner $dir/$y.gradient -|") # gradient times preconditioner.
         all_models+=("nnet-precondition --alpha=$precon_alpha --normalize=true $preconditioner $dir/$y.mdl -|") # model times preconditioner.
         # this is a term that would be there if we had l2 regularization.
      fi
    done
    
    # What we are doing here is doing the combination using the
    # training-data subset and then doing the shrinkage (implemented
    # via nnet-combine-fast with just one input), using the validation
    # data subset.
    $cmd $parallel_opts $dir/log/combine.$x.log \
      nnet-combine-fast --num-threads=$num_threads --verbose=3 --initial-model=0 \
       "${all_models[@]}" ark:$dir/train_subset.egs - \| \
      nnet-combine-fast --num-threads=$num_threads - ark:$dir/valid.egs - \| \
      nnet-am-copy --stats-from=$dir/$x.gradient - - \| \
      nnet-am-fix $fix_opts - $dir/$[$x+1].mdl || exit 1;
  fi
  x=$[$x+1];
done


## Do the final combination using the validation subset.
## Note: don't include the most recent model in the subspace, it wouldn't
## add anything.

preconditioner=$dir/$[x-1].precon
for y in `seq $[x-1] -1 $[$x-$n-1]`; do
  if [ $y -ge $num_sgd_iters ]; then
    all_models+=($dir/$y.mdl) # the model (append to array)
         # Note: we only give the "alpha" argument for didactic reasons.  It won't
         # have an effect, what matters is the alpha value we used above.
    all_models+=("nnet-precondition --alpha=$precon_alpha --normalize=true $preconditioner $dir/$y.gradient -|") # gradient times preconditioner.
    all_models+=("nnet-precondition --alpha=$precon_alpha --normalize=true $preconditioner $dir/$y.mdl -|") # model times preconditioner.
         # this is a term that would be there if we had l2 regularization.
  fi
done

rm $dir/final.mdl 2>/dev/null
$cmd $parallel_opts $dir/log/combine.log \
  nnet-combine-fast --verbose=3 --initial-model=0 \
  "${all_models[@]}" ark:$dir/valid.egs $dir/final.mdl || exit 1;

echo Done
