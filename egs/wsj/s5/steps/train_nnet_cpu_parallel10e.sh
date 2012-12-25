#!/bin/bash

# Copyright 2012  Johns Hopkins University (Author: Daniel Povey).  Apache 2.0.

# 10e is as 10d, but using the -parallel not -simple code, which is
# Hogwild.

# 10d should be basically the same as 10c, but changed slightly so as to
# accommodate some reworked code where nnet-randomize takes posteriors, not
# alignments.

# 10c is as 10b, but adding the ability to mix up, increasing the #neurons in
# the last layer.

# 10b is as 10, but extra parameter apply-shrinking [affects whether
# we apply shrinking on the iterations "between" when we compute the
# shrinking parameters.

# the parallel10 script is as the parallel9 script but doing the shrinking
# in a different way that doesn't lead to instability: after the first few
# iters, we do the shrinking every 3 iters, and on iterations between those,
# apply the shrinking parameters from the most recent shrinking iter.

# the "parallel9" script is as "parallel7" but has the option to add the layers
# a bit more slowly (by skipping iterations), uses a smaller number of
# validation frames for the shrinking on each iteration (but not for the final
# combination), and also, after we're done adding the layers, it will start
# applying the *previous* iteration's shrinking parameters to the current one,
# so that the shrinking can go in parallel with the traiing; this could as much
# as halve training time.

# This "parallel6" script does not attempt to automatically adjust the
# learning rate, but instead modifies it according to a pre-set schedule.

# This parallel5 scripts is modified from the parallel4 scripts, which in turn
# is like the "parallel" script but modifying it so the randomization of frames
# is done in parallel with the training.
#
# This script does a simpler optimization in the "combine" phase, with
# the #params the same as the #layers: for each layer we move along a line
# between the old params, and the average of the new params from the parallel
# SGD.  However, once the objective function improvement on the iteration gets
# below "valid_impr_threshold" (default 0.5), it will "overshoot" by multiplying
# the step direction by the "overshoot" parameter (default 0.8).
# It also updates the learning rates by multiplying them by the step
# lengths worked out above, and this is now done inside the nnet-combine-a program. 
# to the "combine" stage-- one to terminate early, and one to put more debug.


# Begin configuration section.
cmd=run.pl
num_iters=40   # Total number of iterations
num_iters_final=10 # Number of final iterations to give to the
                   # optimization over the validation set.
initial_learning_rate=0.0025
final_learning_rate=0.00025 # 1/10 of the initial one.
num_valid_utts=300    # held-out utterances, used only for diagnostics.
num_valid_frames_shrink=2000 # a subset of the frames in "valid_utts", used only
                             # for estimating shrinkage parameters and for
                             # objective-function reporting.
shrink_interval=3 # Re-compute the shrinkage parameters every 3 iters,
                # except at the start of training when we do it every iter.
apply_shrinking=true
num_valid_frames_combine=10000 # combination weights at the very end.
minibatch_size=1000
minibatches_per_phase=100 # only affects diagnostic messages.
samples_per_iteration=200000 # each iteration of training, see this many samples
                             # per job.
num_jobs_nnet=8 # Number of neural net jobs to run in parallel.

add_layers_period=2 # by default, add new layers every 2 iterations.
num_hidden_layers=2
initial_num_hidden_layers=1  # we'll add the rest one by one.
num_parameters=2000000 # 2 million parameters by default.
stage=-5
realign_iters=""
beam=10  # for realignment.
retry_beam=40
scale_opts="--transition-scale=1.0 --acoustic-scale=0.1 --self-loop-scale=0.1"
parallel_opts=
nnet_config_opts=
splice_width=4 # meaning +- 4 frames on each side for second LDA
lda_dim=250
randprune=4.0 # speeds up LDA.
# If you specify alpha, then we'll do the "preconditioned" update.
alpha=
shrink=true
mix_up=0 # Number of components to mix up to (should be > #tree leaves, if
        # specified.)
num_threads=16
# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;


if [ $# != 4 ]; then
  echo "Usage: steps/train_nnet_cpu_parallel5.sh <data> <lang> <ali-dir> <exp-dir>"
  echo " e.g.: steps/train_nnet_cpu_parallel5.sh data/train_si84 data/lang \\"
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
# in this dir we'll have just one job.
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
     split_feats="ark,s,cs:apply-cmvn --norm-vars=false --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | add-deltas ark:- ark:- |"
    valid_feats="ark,s,cs:utils/filter_scp.pl $dir/valid_uttlist $data/feats.scp | apply-cmvn --norm-vars=false --utt2spk=ark:$data/utt2spk scp:$data/cmvn.scp scp:- ark:- | add-deltas ark:- ark:- |"
   ;;
  lda) feats="ark,s,cs:utils/filter_scp.pl --exclude $dir/valid_uttlist $data/feats.scp | apply-cmvn --norm-vars=false --utt2spk=ark:$data/utt2spk scp:$data/cmvn.scp scp:- ark:- | splice-feats $splice_opts ark:- ark:- | transform-feats $dir/final.mat ark:- ark:- |"
      split_feats="ark,s,cs:apply-cmvn --norm-vars=false --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | splice-feats $splice_opts ark:- ark:- | transform-feats $dir/final.mat ark:- ark:- |"
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
  echo "Creating subset of frames of validation set for shrinking."
  $cmd $dir/log/create_valid_subset_shrink.log \
    nnet-randomize-frames $nnet_context_opts --num-samples=$num_valid_frames_shrink --srand=0 \
       "$valid_feats" "ark,cs:gunzip -c $dir/ali.*.gz | ali-to-pdf $dir/0.mdl ark:- ark:- | ali-to-post ark:- ark:- |" \
     ark:$dir/valid_shrink.egs || exit 1;
  echo "Creating subset of frames of validation set for estimating combination weights."
  $cmd $dir/log/create_valid_subset_combine.log \
    nnet-randomize-frames $nnet_context_opts --num-samples=$num_valid_frames_combine --srand=0 \
       "$valid_feats" "ark,cs:gunzip -c $dir/ali.*.gz | ali-to-pdf $dir/0.mdl ark:- ark:- | ali-to-post ark:- ark:- |" \
     ark:$dir/valid_combine.egs || exit 1;
fi

# up till $last_normal_shrink_iter we will shrink the parameters
# in the normal way using the dev set, but after that we will
# only re-compute the shrinkage parameters periodically.
last_normal_shrink_iter=$[($num_hidden_layers-$initial_num_hidden_layers+1)*$add_layers_period + 2]
mix_up_iter=$last_normal_shrink_iter  # this is pretty arbitrary.

x=-1 # iterations start from 0 but
     # we always start off the "randomization" on the previous iteration.
while [ $x -lt $num_iters ]; do
  # note: archive for aligments won't be sorted as the shell glob "*" expands
  # them in alphabetic not numeric order, so we can't use ark,s,cs: below, only
  # ark,cs which means the features are in sorted order [hence alignments will
  # be called in sorted order (cs).

  wait $last_randomize_process # wait for any randomization from last time.
  y=$[$x+1];
  if [ $stage -le $y ]; then # start off randomization for next time.
    egs_list=
    for n in `seq 1 $num_jobs_nnet`; do
       egs_list="$egs_list ark:$dir/egs.$y.tmp.$n"
    done
    # run the next randomization in the background, so it can run
    # while we're doing the training and combining.
    $cmd $parallel_opts $dir/log/randomize.$y.log \
     nnet-randomize-frames \
        $nnet_context_opts --num-samples=$[$samples_per_iteration*$num_jobs_nnet] \
      --srand=$y "$feats" \
       "ark,cs:gunzip -c $dir/ali.*.gz | ali-to-pdf $alidir/final.mdl ark:- ark:- | ali-to-post ark:- ark:- |" ark:- \| \
       nnet-copy-egs ark:- $egs_list &
    last_randomize_process=$! # process-id of the process what we just spawned.
  fi


  if [ $x -ge 0 ] && [ $stage -le $x ]; then
    # Set off a job that does diagnostics, in the background.
    $cmd $parallel_opts $dir/log/compute_prob.$x.log \
      nnet-compute-prob $dir/$x.mdl ark:$dir/valid_shrink.egs &

    if echo $realign_iters | grep -w $x >/dev/null; then
      wait; # Wait for any "randomize" processes that are running, as
            # these could crash if we change the alignments on disk.   
      echo "Realigning data (pass $x)"
      $cmd JOB=1:$nj $dir/log/align.$x.JOB.log \
        nnet-align-compiled $scale_opts --beam=$beam --retry-beam=$retry_beam "$dir/$x.mdl" \
         "ark:gunzip -c $dir/fsts.JOB.gz|" "$split_feats" \
        "ark:|gzip -c >$dir/ali.JOB.gz" || exit 1;
    fi
    echo "Training neural net (pass $x)"
    if [ $x -gt 0 ] && \
       [ $x -le $[($num_hidden_layers-$initial_num_hidden_layers)*$add_layers_period] ] && \
       [ $[($x-1) % $add_layers_period] -eq 0 ]; then
      mdl="nnet-init --srand=$x $dir/hidden_layer.config - | nnet-insert $dir/$x.mdl - - |"
    else
      mdl=$dir/$x.mdl
    fi
    m=$minibatches_per_phase

    $cmd $parallel_opts JOB=1:$num_jobs_nnet $dir/log/train.$x.JOB.log \
      OMP_NUM_THREADS=1 nnet-train-parallel --num-threads=$num_threads --minibatch-size=$minibatch_size \
        --verbose=2 "$mdl" ark:$dir/egs.$x.tmp.JOB $dir/$[$x+1].JOB.mdl \
       || exit 1;

    egs_list=
    nnets_list=
    for n in `seq 1 $num_jobs_nnet`; do
      egs_list="$egs_list $dir/egs.$x.tmp.$n"
      nnets_list="$nnets_list $dir/$[$x+1].$n.mdl"
    done

    learning_rate=`perl -e '($x,$n,$i,$f)=@ARGV; print $i*exp($x*log($f/$i)/$n);' $[$x+1] $num_iters $initial_learning_rate $final_learning_rate`;

    $cmd $parallel_opts $dir/log/average.$x.log \
       nnet-am-average $nnets_list - \| \
       nnet-am-copy --learning-rate=$learning_rate - $dir/$[$x+1].mdl || exit 1;

    if $shrink; then
      if [ $x -le $last_normal_shrink_iter ] || [ $[$x % $shrink_interval] -eq 0 ]; then
        # For earlier iterations (while we've recently beeen adding layers), or every
        # $shrink_interval=3 iters , just do shrinking normally.
        $cmd $parallel_opts $dir/log/shrink.$x.log \
          nnet-shrink $dir/$[$x+1].mdl ark:$dir/valid_shrink.egs $dir/$[$x+1].mdl || exit 1;
      else
        last_shrink_iter=$[$x - ($x % $shrink_interval)];
        # Get the shrinking parameters from the last shrinking log.
        scales=`grep 'scale factors per layer are' $dir/log/shrink.$last_shrink_iter.log | \
                sed 's/.*\[//' | sed 's/\]//' | perl -ane 'print join(":", split(" ", $_));'`
        [ -z "$scales" ] && echo "Error getting scale factors from log for shrinking" && exit 1;
        if $apply_shrinking; then
           $cmd $dir/log/apply_shrinking.$x.log \
             nnet-am-copy --scales=$scales $dir/$[$x+1].mdl $dir/$[$x+1].mdl || exit 1;
        fi
      fi
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

rm $dir/final.mdl 2>/dev/null

# At the end, final.mdl will be a combination of the last e.g. 10 models.
nnets_list=
for x in `seq $[$num_iters-$num_iters_final+1] $num_iters`; do
  nnets_list="$nnets_list $dir/$x.mdl"
done
$cmd $parallel_opts $dir/log/combine.log \
  nnet-am-combine $nnets_list ark:$dir/valid_combine.egs $dir/final.mdl || exit 1;

# Compute the probability of the final, combined model with
# the same subset we used for the previous compute_probs, as the
# different subsets will lead to different probs.
$cmd $parallel_opts $dir/log/compute_prob.final.log \
  nnet-compute-prob $dir/final.mdl ark:$dir/valid_shrink.egs || exit 1;

echo Done
