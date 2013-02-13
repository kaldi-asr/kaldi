#!/bin/bash

# Copyright 2012  Johns Hopkins University (Author: Daniel Povey).  Apache 2.0.



# Begin configuration section.
cmd=run.pl
num_epochs=15 # Number of epochs during which we reduce
              # the learning rate; number of iteration is worked out from this.
num_epochs_extra=5 # Number of epochs after we stop reducing
                   # the learning rate.
num_iters_final=10 # Number of final iterations to give to the
                   # optimization over the validation set.
initial_learning_rate=0.02 # for RM; or 0.01 is suitable for Swbd.
final_learning_rate=0.004  # for RM; or 0.001 is suitable for Swbd.
num_utts_subset=300    # number of utterances in validation and training
                       # subsets used for shrinkage and diagnostics
num_valid_frames_shrink=0 # number of validation frames in the subset
                             # used for shrinking
num_train_frames_shrink=2000  # number of training frames in the subset used
                              # for shrinking (by default we use all training
                              # frames for this.)
shrink_interval=3 # shrink every $shrink_interval iters,
                # except at the start of training when we do it every iter.
num_valid_frames_combine=0 # #valid frames for combination weights at the very end.
num_train_frames_combine=10000 # # train frames for the above.
num_frames_diagnostic=4000 # number of frames for "compute_prob" jobs
minibatch_size=128 # by default use a smallish minibatch size for neural net training; this controls instability
                   # which would otherwise be a problem with multi-threaded update.  Note:
                   # it also interacts with the "preconditioned" update, so it's not completely cost free.
samples_per_iter=400000 # each iteration of training, see this many samples
                             # per job.  This is just a guideline; it will pick a number
                             # that divides the number of samples in the entire data.
shuffle_buffer_size=5000 # This "buffer_size" variable controls randomization of the samples
                # on each iter.  You could set it to 0 or to a large value for complete
                # randomization, but this would both consume memory and cause spikes in
                # disk I/O.  Smaller is easier on disk and memory but less random.  It's
                # not a huge deal though, as samples are anyway randomized right at the start.
num_jobs_nnet=8 # Number of neural net jobs to run in parallel.

add_layers_period=2 # by default, add new layers every 2 iterations.
num_hidden_layers=2
initial_num_hidden_layers=1  # we'll add the rest one by one.
num_parameters=2000000 # 2 million parameters by default.
stage=-9
realign_iters=""
beam=10  # for realignment.
retry_beam=40
scale_opts="--transition-scale=1.0 --acoustic-scale=0.1 --self-loop-scale=0.1"
parallel_opts="-pe smp 16" # by default we use 16 threads; this lets the queue know.
io_opts="-tc 5" # for jobs with a lot of I/O, limits the number running at one time. 
nnet_config_opts=
splice_width=4 # meaning +- 4 frames on each side for second LDA
lda_dim=250
randprune=4.0 # speeds up LDA.
# If alpha is not set to the empty string, will do the preconditioned update.
alpha=4.0
shrink=true
mix_up=0 # Number of components to mix up to (should be > #tree leaves, if
        # specified.)
num_threads=16

valid_is_heldout=false # For some reason, holding out the validation set from the training set
                       # seems to hurt, so by default we don't do it (i.e. it's included in training)
random_copy=false
cleanup=true
# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;


if [ $# != 4 ]; then
  echo "Usage: steps/train_nnet_cpu.sh [opts] <data> <lang> <ali-dir> <exp-dir>"
  echo " e.g.: steps/train_nnet_cpu.sh data/train data/lang exp/tri3_ali exp/ tri4_nnet"
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
  echo "  --num-parameters <num-parameters|2000000>        # #parameters.  E.g. for 3 hours of data, try 750K parameters;"
  echo "                                                   # for 100 hours of data, try 10M"
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
  echo "  --parallel-opts <opts|\"-pe smp 16\">            # extra options to pass to e.g. queue.pl for processes that"
  echo "                                                   # use multiple threads."
  echo "  --io-opts <opts|\"-tc 10\">                      # Options given to e.g. queue.pl for jobs that do a lot of I/O."
  echo "  --minibatch-size <minibatch-size|128>            # Size of minibatch to process (note: product with --num-threads"
  echo "                                                   # should not get too large, e.g. >2k)."
  echo "  --samples-per-iter <#samples|400000>             # Number of samples of data to process per iteration, per"
  echo "                                                   # process."
  echo "  --splice-width <width|4>                         # Number of frames on each side to append for feature input"
  echo "                                                   # (note: we splice processed, typically 40-dimensional frames"
  echo "  --lda-dim <dim|250>                              # Dimension to reduce spliced features to with LDA"
  echo "  --num-iters-final <#iters|10>                    # Number of final iterations to give to nnet-combine-fast to "
  echo "                                                   # interpolate parameters (the weights are learned with a validation set)"
  echo "  --num-utts-subset <#utts|300>                    # Number of utterances in subsets used for validation and diagnostics"
  echo "                                                   # (the validation subset is held out from training)"
  echo "  --num-valid-frames-shrink <#frames|2000>         # Number of frames from the validation set used for shrinking"
  echo "  --num-train-frames-shrink <#frames|0>            # Number of frames from the training set used for shrinking"
  echo "                                                   # (by default it's included in training, which for some reason helps)."
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
awk '{print $1}' $data/utt2spk | utils/shuffle_list.pl | head -$num_utts_subset \
    > $dir/valid_uttlist || exit 1;
awk '{print $1}' $data/utt2spk | utils/filter_scp.pl --exclude $dir/valid_uttlist | \
     head -$num_utts_subset > $dir/train_subset_uttlist || exit 1;


## Set up features.  Note: these are different from the normal features
## because we have one rspecifier that has the features for the entire
## training set, not separate ones for each batch.
if [ -f $alidir/final.mat ]; then feat_type=lda; else feat_type=delta; fi
echo "$0: feature type is $feat_type"

case $feat_type in
  delta) feats="ark,s,cs:utils/filter_scp.pl --exclude $dir/valid_uttlist $sdata/JOB/feats.scp | apply-cmvn --norm-vars=false --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:- ark:- | add-deltas ark:- ark:- |"
    valid_feats="ark,s,cs:utils/filter_scp.pl $dir/valid_uttlist $data/feats.scp | apply-cmvn --norm-vars=false --utt2spk=ark:$data/utt2spk scp:$data/cmvn.scp scp:- ark:- | add-deltas ark:- ark:- |"
    train_subset_feats="ark,s,cs:utils/filter_scp.pl $dir/train_subset_uttlist $data/feats.scp | apply-cmvn --norm-vars=false --utt2spk=ark:$data/utt2spk scp:$data/cmvn.scp scp:- ark:- | add-deltas ark:- ark:- |"
   ;;
  lda) feats="ark,s,cs:utils/filter_scp.pl --exclude $dir/valid_uttlist $sdata/JOB/feats.scp | apply-cmvn --norm-vars=false --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:- ark:- | splice-feats $splice_opts ark:- ark:- | transform-feats $dir/final.mat ark:- ark:- |"
      valid_feats="ark,s,cs:utils/filter_scp.pl $dir/valid_uttlist $data/feats.scp | apply-cmvn --norm-vars=false --utt2spk=ark:$data/utt2spk scp:$data/cmvn.scp scp:- ark:- | splice-feats $splice_opts ark:- ark:- | transform-feats $dir/final.mat ark:- ark:- |"
      train_subset_feats="ark,s,cs:utils/filter_scp.pl $dir/train_subset_uttlist $data/feats.scp | apply-cmvn --norm-vars=false --utt2spk=ark:$data/utt2spk scp:$data/cmvn.scp scp:- ark:- | splice-feats $splice_opts ark:- ark:- | transform-feats $dir/final.mat ark:- ark:- |"
    cp $alidir/final.mat $dir    
    ;;
  *) echo "$0: invalid feature type $feat_type" && exit 1;
esac
if [ -f $alidir/trans.1 ]; then
  echo "$0: using transforms from $alidir"
  feats="$feats transform-feats --utt2spk=ark:$sdata/JOB/utt2spk ark:$alidir/trans.JOB ark:- ark:- |"
  valid_feats="$valid_feats transform-feats --utt2spk=ark:$data/utt2spk 'ark:cat $alidir/trans.*|' ark:- ark:- |"
  train_subset_feats="$train_subset_feats transform-feats --utt2spk=ark:$data/utt2spk 'ark:cat $alidir/trans.*|' ark:- ark:- |"
fi

if [ $stage -le -9 ]; then
  echo "$0: working out number of frames of training data"
  num_frames=`feat-to-len scp:$data/feats.scp ark,t:- | awk '{x += $2;} END{print x;}'` || exit 1;
  echo $num_frames > $dir/num_frames
else
  num_frames=`cat $dir/num_frames` || exit 1;
fi

# Working out number of iterations per epoch.
iters_per_epoch=`perl -e "print int($num_frames/($samples_per_iter * $num_jobs_nnet) + 0.5);"` || exit 1;
[ $iters_per_epoch -eq 0 ] && iters_per_epoch=1
samples_per_iter_real=$[$num_frames/($num_jobs_nnet*$iters_per_epoch)]
echo "Every epoch, splitting the data up into $iters_per_epoch iterations,"
echo "giving samples-per-iteration of $samples_per_iter_real (you requested $samples_per_iter)."


## Do LDA on top of whatever features we already have; store the matrix which
## we'll put into the neural network as a constant.

if [ $stage -le -8 ]; then
  echo "$0: Accumulating LDA statistics."
  $cmd JOB=1:$nj $dir/log/lda_acc.JOB.log \
    ali-to-post "ark:gunzip -c $alidir/ali.JOB.gz|" ark:- \| \
      weight-silence-post 0.0 $silphonelist $alidir/final.mdl ark:- ark:- \| \
      acc-lda --rand-prune=$randprune $alidir/final.mdl "$feats splice-feats --left-context=$splice_width --right-context=$splice_width ark:- ark:- |" ark,s,cs:- \
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


if [ $stage -le -7 ]; then
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

if [ $stage -le -6 ]; then
  echo "Training transition probabilities and setting priors"
  $cmd $dir/log/train_trans.log \
    nnet-train-transitions $dir/0.mdl "ark:gunzip -c $alidir/ali.*.gz|" $dir/0.mdl \
    || exit 1;
fi

if [ $stage -le -5 ]; then
  echo "Compiling graphs of transcripts"
  $cmd JOB=1:$nj $dir/log/compile_graphs.JOB.log \
    compile-train-graphs $dir/tree $dir/0.mdl  $lang/L.fst  \
     "ark:utils/sym2int.pl --map-oov $oov -f 2- $lang/words.txt < $data/split$nj/JOB/text |" \
      "ark:|gzip -c >$dir/fsts.JOB.gz" || exit 1;
fi

cp $alidir/ali.*.gz $dir


nnet_context_opts="--left-context=`nnet-am-info $dir/0.mdl 2>/dev/null | grep -w left-context | awk '{print $2}'` --right-context=`nnet-am-info $dir/0.mdl 2>/dev/null | grep -w right-context | awk '{print $2}'`" || exit 1;

if [ $stage -le -4 ]; then
  echo "Getting validation and training subset examples."
  rm $dir/.error 2>/dev/null
  $cmd $dir/log/create_valid_subset.log \
    nnet-get-egs $nnet_context_opts "$valid_feats" \
     "ark,cs:gunzip -c $dir/ali.*.gz | ali-to-pdf $dir/0.mdl ark:- ark:- | ali-to-post ark:- ark:- |" \
     "ark:$dir/valid_all.egs" || touch $dir/.error &
  $cmd $dir/log/create_train_subset.log \
    nnet-get-egs $nnet_context_opts "$train_subset_feats" \
     "ark,cs:gunzip -c $dir/ali.*.gz | ali-to-pdf $dir/0.mdl ark:- ark:- | ali-to-post ark:- ark:- |" \
     "ark:$dir/train_subset_all.egs" || touch $dir/.error &
  wait;
  [ -f $dir/.error ] && exit 1;
  echo "Getting subsets of validation examples for shrinking, diagnostics and combination."
  $cmd $dir/log/create_valid_subset_shrink.log \
    nnet-subset-egs --n=$num_valid_frames_shrink ark:$dir/valid_all.egs \
     ark:$dir/valid_shrink.egs || touch $dir/.error &
  $cmd $dir/log/create_valid_subset_combine.log \
    nnet-subset-egs --n=$num_valid_frames_combine ark:$dir/valid_all.egs \
        ark:$dir/valid_combine.egs || touch $dir/.error &
  $cmd $dir/log/create_valid_subset_diagnostic.log \
    nnet-subset-egs --n=$num_frames_diagnostic ark:$dir/valid_all.egs \
    ark:$dir/valid_diagnostic.egs || touch $dir/.error &

  $cmd $dir/log/create_train_subset_shrink.log \
    nnet-subset-egs --n=$num_train_frames_shrink ark:$dir/train_subset_all.egs \
    ark:$dir/train_shrink.egs || touch $dir/.error &
  $cmd $dir/log/create_train_subset_combine.log \
    nnet-subset-egs --n=$num_train_frames_combine ark:$dir/train_subset_all.egs \
    ark:$dir/train_combine.egs || touch $dir/.error &
  $cmd $dir/log/create_train_subset_diagnostic.log \
    nnet-subset-egs --n=$num_frames_diagnostic ark:$dir/train_subset_all.egs \
    ark:$dir/train_diagnostic.egs || touch $dir/.error &
  wait
  cat $dir/valid_shrink.egs $dir/train_shrink.egs > $dir/shrink.egs
  cat $dir/valid_combine.egs $dir/train_combine.egs > $dir/combine.egs

  for f in $dir/{shrink,combine,train_diagnostic,valid_diagnostic}.egs; do
    [ ! -s $f ] && echo "No examples in file $f" && exit 1;
  done
  rm $dir/valid_all.egs $dir/train_subset_all.egs $dir/{train,valid}_{shrink,combine}.egs
fi

if [ $stage -le -3 ]; then
  mkdir -p $dir/egs
  mkdir -p $dir/temp
  echo "Creating training examples";
  # in $dir/egs, create $num_jobs_nnet separate files with training examples.
  # The order is not randomized at this point.

  egs_list=
  for n in `seq 1 $num_jobs_nnet`; do
    egs_list="$egs_list ark:$dir/egs/egs_orig.$n.JOB.ark"
  done
  echo "Generating training examples on disk"
  # The examples will go round-robin to egs_list.
  $cmd $io_opts JOB=1:$nj $dir/log/get_egs.JOB.log \
    nnet-get-egs $nnet_context_opts "$feats" \
    "ark,cs:gunzip -c $dir/ali.JOB.gz | ali-to-pdf $alidir/final.mdl ark:- ark:- | ali-to-post ark:- ark:- |" ark:- \| \
    nnet-copy-egs ark:- $egs_list || exit 1;
fi

if [ $stage -le -2 ]; then
  # combine all the "egs_orig.JOB.*.scp" (over the $nj splits of the data) and
  # then split into multiple parts egs.JOB.*.scp for different parts of the
  # data, 0 .. $iters_per_epoch-1.

  if [ $iters_per_epoch -eq 1 ]; then
    echo "Since iters-per-epoch == 1, just concatenating the data."
    for n in `seq 1 $num_jobs_nnet`; do
      cat $dir/egs/egs_orig.$n.*.ark > $dir/egs/egs_tmp.$n.0.ark || exit 1;
      rm $dir/egs/egs_orig.$n.*.ark || exit 1;
    done
  else # We'll have to split it up using nnet-copy-egs.
    egs_list=
    for n in `seq 0 $[$iters_per_epoch-1]`; do
      egs_list="$egs_list ark:$dir/egs/egs_tmp.JOB.$n.ark"
    done
    $cmd $io_opts JOB=1:$num_jobs_nnet $dir/log/split_egs.JOB.log \
      nnet-copy-egs --random=$random_copy --srand=JOB \
        "ark:cat $dir/egs/egs_orig.JOB.*.ark|" $egs_list '&&' \
        rm $dir/egs/egs_orig.JOB.*.ark || exit 1;
  fi
fi

if [ $stage -le -1 ]; then
  # Next, shuffle the order of the examples in each of those files.
  # Each one should not be too large, so we can do this in memory.
  echo "Shuffling the order of training examples"
  echo "(in order to avoid stressing the disk, these won't all run at once)."

  for n in `seq 0 $[$iters_per_epoch-1]`; do
    $cmd $io_opts JOB=1:$num_jobs_nnet $dir/log/shuffle.$n.JOB.log \
      nnet-shuffle-egs "--srand=\$[JOB+($num_jobs_nnet*$n)]" \
      ark:$dir/egs/egs_tmp.JOB.$n.ark ark:$dir/egs/egs.JOB.$n.ark '&&' \
      rm $dir/egs/egs_tmp.JOB.$n.ark || exit 1;
  done
fi

num_iters_reduce=$[$num_epochs * $iters_per_epoch];
num_iters_extra=$[$num_epochs_extra * $iters_per_epoch];
num_iters=$[$num_iters_reduce+$num_iters_extra]

echo "Will train for $num_epochs + $num_epochs_extra epochs, equalling "
echo " $num_iters_reduce + $num_iters_extra = $num_iters iterations, "
echo " (while reducing learning rate) + (with constant learning rate)."

# up till $last_normal_shrink_iter we will shrink the parameters
# in the normal way using the dev set, but after that we will
# only re-compute the shrinkage parameters periodically.
last_normal_shrink_iter=$[($num_hidden_layers-$initial_num_hidden_layers+1)*$add_layers_period + 2]
mix_up_iter=$last_normal_shrink_iter  # this is pretty arbitrary.

x=0
while [ $x -lt $num_iters ]; do
  if [ $x -ge 0 ] && [ $stage -le $x ]; then

    # Set off jobs doing some diagnostics, in the background.
    $cmd $dir/log/compute_prob_valid.$x.log \
      nnet-compute-prob $dir/$x.mdl ark:$dir/valid_diagnostic.egs &
    $cmd $dir/log/compute_prob_train.$x.log \
      nnet-compute-prob $dir/$x.mdl ark:$dir/train_diagnostic.egs &

    if echo $realign_iters | grep -w $x >/dev/null; then
      echo "Realigning data (pass $x)"
      $cmd JOB=1:$nj $dir/log/align.$x.JOB.log \
        nnet-align-compiled $scale_opts --beam=$beam --retry-beam=$retry_beam "$dir/$x.mdl" \
         "ark:gunzip -c $dir/fsts.JOB.gz|" "$feats" \
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

    $cmd $parallel_opts JOB=1:$num_jobs_nnet $dir/log/train.$x.JOB.log \
       nnet-shuffle-egs --buffer-size=$shuffle_buffer_size --srand=$x \
         ark:$dir/egs/egs.JOB.$[$x%$iters_per_epoch].ark ark:- \| \
       nnet-train-parallel --num-threads=$num_threads --minibatch-size=$minibatch_size \
        "$mdl" ark:- $dir/$[$x+1].JOB.mdl \
       || exit 1;

    nnets_list=
    for n in `seq 1 $num_jobs_nnet`; do
      nnets_list="$nnets_list $dir/$[$x+1].$n.mdl"
    done

    learning_rate=`perl -e '($x,$n,$i,$f)=@ARGV; print ($x >= $n ? $f : $i*exp($x*log($f/$i)/$n));' $[$x+1] $num_iters_reduce $initial_learning_rate $final_learning_rate`;

    $cmd $dir/log/average.$x.log \
       nnet-am-average $nnets_list - \| \
       nnet-am-copy --learning-rate=$learning_rate - $dir/$[$x+1].mdl || exit 1;

    if $shrink; then
      if [ $x -le $last_normal_shrink_iter ] || [ $[$x % $shrink_interval] -eq 0 ]; then
        # For earlier iterations (while we've recently beeen adding layers), or every
        # $shrink_interval=3 iters , just do shrinking normally.
        mb=$[($num_valid_frames_shrink+$num_train_frames_shrink+$num_threads-1)/$num_threads]
        $cmd $parallel_opts $dir/log/shrink.$x.log \
          nnet-combine-fast --num-threads=$num_threads --verbose=3 --minibatch-size=$mb \
            $dir/$[$x+1].mdl ark:$dir/shrink.egs $dir/$[$x+1].mdl || exit 1;
      fi
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

rm $dir/final.mdl 2>/dev/null

# At the end, final.mdl will be a combination of the last e.g. 10 models.
nnets_list=
for x in `seq $[$num_iters-$num_iters_final+1] $num_iters`; do
  [ $x -gt $mix_up_iter ] && nnets_list="$nnets_list $dir/$x.mdl"
done
if [ $stage -le $num_iters ]; then
  mb=$[($num_valid_frames_combine+$num_train_frames_combine+$num_threads-1)/$num_threads]
  $cmd $parallel_opts $dir/log/combine.log \
    nnet-combine-fast --num-threads=$num_threads --verbose=3 --minibatch-size=$mb \
    $nnets_list ark:$dir/combine.egs $dir/final.mdl || exit 1;
fi

# Compute the probability of the final, combined model with
# the same subset we used for the previous compute_probs, as the
# different subsets will lead to different probs.
$cmd $dir/log/compute_prob_valid.final.log \
  nnet-compute-prob $dir/final.mdl ark:$dir/valid_diagnostic.egs &
$cmd $dir/log/compute_prob_train.final.log \
  nnet-compute-prob $dir/final.mdl ark:$dir/train_diagnostic.egs &

echo Done

if $cleanup; then
  echo Cleaning up data
  echo Removing training examples
  rm -r $dir/egs
  echo Removing most of the models
  for x in `seq 0 $num_iters`; do
    if [ $[$x%10] -ne 0 ] && [ $x -lt $[$num_iters-$num_iters_final+1] ]; then 
       # delete all but every 10th model; don't delete the ones which combine to form the final model.
      rm $dir/$x.mdl
    fi
  done
fi
