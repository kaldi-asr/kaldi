#!/bin/bash   

# Copyright 2012  Johns Hopkins University (Author: Daniel Povey).  Apache 2.0.


# This is as train_nnet_cpu.sh but supports convolutional-in-time
# approaches where at different layers we see temporal context.
# I am also taking the opportunity to remove some un-needed features
# such as shrinking (no longer necessary for ReLUs).

# Begin configuration section.
cmd=run.pl 
num_epochs_per_eon=10 # Number of epochs per LDA stage.
num_epochs_extra=5      # Number of epochs after we stop reducing
                        # the learning rate (after all stages)
num_iters_final=10    # Number of final iterations to give to the
                      # optimization over the validation set.
num_iters_combine=20 # Maximum number of iterations we may try to combine over.
                     # Number used will be the minimum of this and num_iters_extra,
                     # which is itself a function of num_epochs_extra.
initial_learning_rate=0.02 # for RM; or 0.01 is suitable for Swbd.
final_learning_rate=0.004  # for RM; or 0.001 is suitable for Swbd.
old_layer_learning_rate=  # If not set, defaults to final_learning_rate.
final_layer_variance=1.0 # factor on variance for last layer... suggest 0.1 or 0.0..
num_utts_subset=300    # number of utterances in validation and training
                       # subsets used for diagnostics and combination.
within_class_factor=1.0 # affects LDA via scaling of the output (e.g. try setting to 0.01).
num_valid_frames_combine=0 # #valid frames for combination weights at the very end.
num_train_frames_combine=10000 # # train frames for the above.
num_frames_diagnostic=4000 # number of frames for "compute_prob" jobs
minibatch_size=128 # by default use a smallish minibatch size for neural net training; this controls instability
                   # which would otherwise be a problem with multi-threaded update.  Note:
                   # it also interacts with the "preconditioned" update, so it's not completely cost free.
samples_per_iter=400000  # each iteration of training, see this many samples
                         # per job.  This is just a guideline; it will pick a number
                         # that divides the number of samples in the entire data.
shuffle_buffer_size=5000 # This "buffer_size" variable controls randomization of the samples
                # on each iter.  You could set it to 0 or to a large value for complete
                # randomization, but this would both consume memory and cause spikes in
                # disk I/O.  Smaller is easier on disk and memory but less random.  It's
                # not a huge deal though, as samples are anyway randomized right at the start.
num_jobs_nnet=16 # Number of neural net jobs to run in parallel; you need to
                 # keep this in sync with parallel_opts.
feat_type=
initial_dropout_scale=
final_dropout_scale=

add_layers_period=2 # by default, add new layers every 2 iterations.
num_eons=2   # Number of stages of training; each time we do splice + LDA.
               # One LDA on the initial spliced features; then one on the
               # intermediate neural net features.
num_hidden_layers_per_eon=2 # This is the number of full-size hidden layers per eon,
                            # not counting the small one of dimensino $pre_splice_dim.
splice_context=2 # meaning +- 2 frames on each side each time we do
               # an LDA.
pre_splice_dim=100 # Dimension we reduce to before each splicing and LDA.

# LDA options...
randprune=4.0 # speeds up LDA accumulation.

num_parameters=2000000 # 2 million parameters by default.
stage=-9
realign=true # set to false if you don't want to do realignment.
beam=10  # for realignment.
retry_beam=40
scale_opts="--transition-scale=1.0 --acoustic-scale=0.1 --self-loop-scale=0.1"
parallel_opts="-pe smp 16" # by default we use 16 threads; this just lets the queue know.
io_opts="-tc 5" # for jobs with a lot of I/O, limits the number running at one time. 

# If alpha is not set to the empty string, will do the preconditioned update.
alpha=4.0
max_change=10.0 # max parameter-change per minibatch, helps ensure stability.
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
  echo "Usage: steps/train_nnet_cpu_conv.sh [opts] <data> <lang> <ali-dir> <exp-dir>"
  echo " e.g.: steps/train_nnet_cpu_conv.sh data/train data/lang exp/tri3_ali exp/ tri4_nnet"
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
num_leaves=`gmm-info $alidir/final.mdl 2>/dev/null | awk '/number of pdfs/{print $NF}'` || exit 1;
silphonelist=`cat $lang/phones/silence.csl` || exit 1;

nj=`cat $alidir/num_jobs` || exit 1;  # number of jobs in alignment dir...
# in this dir we'll have just one job.
sdata=$data/split$nj
utils/split_data.sh $data $nj

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
if [ -z $feat_type ]; then
  if [ -f $alidir/final.mat ]; then feat_type=lda; else feat_type=delta; fi
fi
echo "$0: feature type is $feat_type"

case $feat_type in
  delta) feats="ark,s,cs:utils/filter_scp.pl --exclude $dir/valid_uttlist $sdata/JOB/feats.scp | apply-cmvn --norm-vars=false --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:- ark:- | add-deltas ark:- ark:- |"
    valid_feats="ark,s,cs:utils/filter_scp.pl $dir/valid_uttlist $data/feats.scp | apply-cmvn --norm-vars=false --utt2spk=ark:$data/utt2spk scp:$data/cmvn.scp scp:- ark:- | add-deltas ark:- ark:- |"
    train_subset_feats="ark,s,cs:utils/filter_scp.pl $dir/train_subset_uttlist $data/feats.scp | apply-cmvn --norm-vars=false --utt2spk=ark:$data/utt2spk scp:$data/cmvn.scp scp:- ark:- | add-deltas ark:- ark:- |"
   ;;
  raw) feats="ark,s,cs:utils/filter_scp.pl --exclude $dir/valid_uttlist $sdata/JOB/feats.scp | apply-cmvn --norm-vars=false --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:- ark:- |"
    valid_feats="ark,s,cs:utils/filter_scp.pl $dir/valid_uttlist $data/feats.scp | apply-cmvn --norm-vars=false --utt2spk=ark:$data/utt2spk scp:$data/cmvn.scp scp:- ark:- |"
    train_subset_feats="ark,s,cs:utils/filter_scp.pl $dir/train_subset_uttlist $data/feats.scp | apply-cmvn --norm-vars=false --utt2spk=ark:$data/utt2spk scp:$data/cmvn.scp scp:- ark:- |"
   ;;
  lda) feats="ark,s,cs:utils/filter_scp.pl --exclude $dir/valid_uttlist $sdata/JOB/feats.scp | apply-cmvn --norm-vars=false --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:- ark:- | splice-feats $splice_opts ark:- ark:- | transform-feats $dir/final.mat ark:- ark:- |"
      valid_feats="ark,s,cs:utils/filter_scp.pl $dir/valid_uttlist $data/feats.scp | apply-cmvn --norm-vars=false --utt2spk=ark:$data/utt2spk scp:$data/cmvn.scp scp:- ark:- | splice-feats $splice_opts ark:- ark:- | transform-feats $dir/final.mat ark:- ark:- |"
      train_subset_feats="ark,s,cs:utils/filter_scp.pl $dir/train_subset_uttlist $data/feats.scp | apply-cmvn --norm-vars=false --utt2spk=ark:$data/utt2spk scp:$data/cmvn.scp scp:- ark:- | splice-feats $splice_opts ark:- ark:- | transform-feats $dir/final.mat ark:- ark:- |"
    cp $alidir/final.mat $dir    
    ;;
  *) echo "$0: invalid feature type $feat_type" && exit 1;
esac
if [ -f $alidir/trans.1 ] && [ $feat_type != "raw" ]; then
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
  ! num_frames=`cat $dir/num_frames` && \
    echo "file $dir/num_frames does not exist: perhaps running with invalid --stage option?" && \
    exit 1;
fi

# Working out number of iterations per epoch.
iters_per_epoch=`perl -e "print int($num_frames/($samples_per_iter * $num_jobs_nnet) + 0.5);"` || exit 1;
[ $iters_per_epoch -eq 0 ] && iters_per_epoch=1
samples_per_iter_real=$[$num_frames/($num_jobs_nnet*$iters_per_epoch)]
echo "Every epoch, splitting the data up into $iters_per_epoch iterations,"
echo "giving samples-per-iteration of $samples_per_iter_real (you requested $samples_per_iter)."


feat_dim=`feat-to-dim "$valid_feats" -` || exit 1;

# Working out hidden-layer size [not counting the LDA parameters as being 
# parameters, as they're not trainable in the net.]  Dimensions of input, intermediate,
# output features are as follows, if 
#                  h is  hidden-layer dimension (variable we are solving for)
#                  n is (splice_context * 2 + 1)
#                  d is input feature dim.
#                  p is pre_splice_dim, which is small-ish dimension we create prior to the 
#                       output layer each time we prepare to do the "intermediate" LDA.
#            num-pdfs is the number of pdfs in the system.
#                 Assume for this diagram that we have two full-size hidden layers between
#                 each splice+LDA, and two LDA stages.
#   d [splice]-> (n * d) [lda]-> (n * d) -> h -> h -> p -> [splice]-> (n * p) [lda]-> (n * p) -> h -> h -> num-pdfs
#
# The number of trainable parameters (not counting lda-type transforms) is:
#   (n * d) * h +
#   h * (num_eons - 1) * (n * p) +
#   h * h * (num_hidden_layers_per_eon - 1) * num_eons +
#   h * num_pdfs
# which we can write as a 2nd order polynomial in h, equate to the
# number of parameters, and arrange as:
#   a h^2 + b h + c = 0 , with
#   a = ((num_hidden_layers_per_eon - 1) * num_eons)
#   b = ((n * d) + ((num_eons - 1) * (n * p)) + num_pdfs), 
#   c = -num_parameters,
#  so we get
#  h =  (-b + sqrt(b^2 - 4 a c)) / (2a)

num_splice=`echo $[2*$splice_context + 1]`;
num_pdfs=`tree-info $dir/tree | grep num-pdfs | awk '{print $2;}'`
echo "$0: Number of pdfs is $num_pdfs"

hidden_layer_size=`perl -we '($num_parameters,$feat_dim,$num_eons,$num_hidden_layers_per_eon,$n,$p,$num_pdfs) = @ARGV;
     $a = (($num_hidden_layers_per_eon - 1) * $num_eons);
     $b = ($n * $feat_dim) + ($num_eons - 1) * ($n * $p) + $num_pdfs;
     $c = -$num_parameters;
     if ($a != 0.0) {  $h = int((-$b + sqrt($b*$b - 4 * $a * $c)) / (2*$a)); }
     else { $h = int(-$c / $b); }
     print $h;' $num_parameters $feat_dim $num_eons $num_hidden_layers_per_eon $num_splice $pre_splice_dim $num_pdfs` || exit 1;

! [ $hidden_layer_size -gt 0 ] && exit 1;

echo "$0: Hidden layer size is $hidden_layer_size"

## Do LDA on top of whatever features we already have; store the matrix which
## we'll put into the neural network as a constant.

if [ $stage -le -8 ]; then
  echo "$0: Accumulating LDA statistics."
  $cmd JOB=1:$nj $dir/log/lda_acc.JOB.log \
    ali-to-post "ark:gunzip -c $alidir/ali.JOB.gz|" ark:- \| \
      weight-silence-post 0.0 $silphonelist $alidir/final.mdl ark:- ark:- \| \
      acc-lda --rand-prune=$randprune $alidir/final.mdl "$feats splice-feats --left-context=$splice_context --right-context=$splice_context ark:- ark:- |" ark,s,cs:- \
       $dir/lda.JOB.acc || exit 1;

  lda_dim=$[$feat_dim*$num_splice]; # We do LDA without dimension reduction;
             # it's a special form of preconditioning.
  est-lda --allow-large-dim=true --within-class-factor=$within_class_factor --dim=$lda_dim $dir/lda.mat $dir/lda.*.acc \
      2>$dir/log/lda_est.log || exit 1;
  rm $dir/lda.*.acc
  echo "Computed LDA"
fi


if [ $stage -le -7 ]; then
  echo "$0: initializing neural net";
  ## Initialize a neural-net config with one hidden layer and
  ## the computed LDA matrix.

  spliced_dim=$[$feat_dim*$num_splice]
  param_stddev=`perl -e "print 1.0/sqrt($spliced_dim);"`
  cat > $dir/nnet.config <<EOF
SpliceComponent input-dim=$feat_dim left-context=$splice_context right-context=$splice_context
FixedLinearComponent matrix=$dir/lda.mat
AffineComponentPreconditioned input-dim=$spliced_dim output-dim=$hidden_layer_size alpha=$alpha max-change=$max_change learning-rate=$initial_learning_rate param-stddev=$param_stddev bias-stddev=0
RectifiedLinearComponent dim=$hidden_layer_size
AffineComponentPreconditioned input-dim=$hidden_layer_size output-dim=$num_pdfs alpha=$alpha max-change=$max_change learning-rate=$initial_learning_rate param-stddev=0 bias-stddev=0
SoftmaxComponent dim=$num_pdfs
EOF
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

if [ $stage -le -5 ] && $realign; then
  echo "Compiling graphs of transcripts"
  $cmd JOB=1:$nj $dir/log/compile_graphs.JOB.log \
    compile-train-graphs $dir/tree $dir/0.mdl  $lang/L.fst  \
     "ark:utils/sym2int.pl --map-oov $oov -f 2- $lang/words.txt < $data/split$nj/JOB/text |" \
      "ark:|gzip -c >$dir/fsts.JOB.gz" || exit 1;
fi

cp $alidir/ali.*.gz $dir


full_context=$[$splice_context*$num_eons] || exit 1;
nnet_context_opts="--left-context=$full_context --right-context=$full_context"

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
  echo "Getting subsets of validation examples for diagnostics and combination."
  $cmd $dir/log/create_valid_subset_combine.log \
    nnet-subset-egs --n=$num_valid_frames_combine ark:$dir/valid_all.egs \
        ark:$dir/valid_combine.egs || touch $dir/.error &
  $cmd $dir/log/create_valid_subset_diagnostic.log \
    nnet-subset-egs --n=$num_frames_diagnostic ark:$dir/valid_all.egs \
    ark:$dir/valid_diagnostic.egs || touch $dir/.error &

  $cmd $dir/log/create_train_subset_combine.log \
    nnet-subset-egs --n=$num_train_frames_combine ark:$dir/train_subset_all.egs \
    ark:$dir/train_combine.egs || touch $dir/.error &
  $cmd $dir/log/create_train_subset_diagnostic.log \
    nnet-subset-egs --n=$num_frames_diagnostic ark:$dir/train_subset_all.egs \
    ark:$dir/train_diagnostic.egs || touch $dir/.error &
  wait
  cat $dir/valid_combine.egs $dir/train_combine.egs > $dir/combine.egs

  for f in $dir/{combine,train_diagnostic,valid_diagnostic}.egs; do
    [ ! -s $f ] && echo "No examples in file $f" && exit 1;
  done
  rm $dir/valid_all.egs $dir/train_subset_all.egs $dir/{train,valid}_combine.egs
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

num_iters_per_eon=$[$num_epochs_per_eon * $iters_per_epoch];
num_iters_extra=$[$num_epochs_extra * $iters_per_epoch];
num_iters=$[$num_iters_per_eon*$num_eons + $num_iters_extra]
[ -z "$old_layer_learning_rate" ] && old_layer_learning_rate=$final_learning_rate

echo "Will train for $num_iters total iterations: $num_iters_per_eon per eon times $num_eons eons, plus $num_iters_extra iters at the end"


# Get the iteration number on which we'll mix up. [Don't do this until
# we've added the last
mix_up_iter_of_last_eon=$[($num_hidden_layers_per_eon-1)*$add_layers_period + 2]
mix_up_iter=$[$mix_up_iter_of_last_eon + $num_iters_per_eon*($num_eons-1)]


function do_eon_start_computation {
  # Called at the start of an eon (but not the 1st eon)
  echo "Preparing to do LDA computation at the start of eon $eon"
  
  echo "Doing SVD on final layer"
  $cmd $dir/log/limit_rank_final.$y.log \
    nnet-am-limit-rank-final --dim=$pre_splice_dim $dir/$y.mdl $dir/temp.mdl || exit 1;
  
  # Get the #components in this model.
  num_components=`nnet-am-info $dir/temp.mdl | grep num-components | awk '{print $2}'`
  
  # First we extract the raw neural net, with the last two components (the softmax
  # layer and the affine transform that precedes it) removed.  We put in "raw.$eon.mdl" the
  # raw neural net.
  nnet-am-copy --learning-rate=$old_layer_learning_rate $dir/temp.mdl - | \
     nnet-to-raw-nnet --truncate=$[$num_components-2] - $dir/raw.$eon.net

  nnet_feats="$feats nnet-compute $dir/raw.$eon.net ark:- ark:- | splice-feats --left-context=$splice_context --right-context=$splice_context ark:- ark:- |"
  
  echo "$0: Accumulating LDA statistics for eon $eon."
  $cmd JOB=1:$nj $dir/log/lda_acc_eon$eon.JOB.log \
    ali-to-post "ark:gunzip -c $alidir/ali.JOB.gz|" ark:- \| \
    weight-silence-post 0.0 $silphonelist $alidir/final.mdl ark:- ark:- \| \
    acc-lda --rand-prune=$randprune $alidir/final.mdl "$nnet_feats" ark,s,cs:- \
    $dir/lda.JOB.acc || exit 1;

  lda_dim=$[$pre_splice_dim*$num_splice]; # We do LDA without dimension reduction;
                     # it's a special form of preconditioning.

  echo "$0: estimating LDA for eon $eon"
  nnet-get-feature-transform --allow-large-dim=true --within-class-factor=$within_class_factor --dim=$lda_dim $dir/lda.$eon.mat $dir/lda.*.acc \
    2>$dir/log/lda_est_eon$eon.log || exit 1;
  rm $dir/lda.*.acc
  
  # Create last few layers of the nnet, to be appended to raw.$eon.net
  param_stddev_hidden=`perl -e "print 1.0/sqrt($lda_dim);"`
  param_stddev_final=`perl -e "print $final_layer_variance/sqrt($num_pdfs);"`
  cat <<EOF > $dir/extra_layers.$eon.config
SpliceComponent input-dim=$pre_splice_dim left-context=$splice_context right-context=$splice_context
FixedAffineComponent matrix=$dir/lda.$eon.mat
AffineComponentPreconditioned input-dim=$lda_dim output-dim=$hidden_layer_size alpha=$alpha max-change=$max_change learning-rate=$initial_learning_rate param-stddev=$param_stddev_hidden bias-stddev=0
RectifiedLinearComponent dim=$hidden_layer_size
AffineComponentPreconditioned input-dim=$hidden_layer_size output-dim=$num_pdfs alpha=$alpha max-change=$max_change learning-rate=$initial_learning_rate param-stddev=$param_stddev_final bias-stddev=0
SoftmaxComponent dim=$num_pdfs
EOF
  $cmd $dir/log/init_nnet.$eon.log \
    nnet-init $dir/extra_layers.$eon.config $dir/raw2.$eon.net || exit 1
  
  $cmd $dir/log/nnet_init.log \
    nnet-am-init $alidir/tree $lang/topo "raw-nnet-concat $dir/raw.$eon.net $dir/raw2.$eon.net -|" \
    $dir/$y.mod.mdl || exit 1;

  echo "Training transition probabilities and setting priors for new eon"
  $cmd $dir/log/train_trans.$eon.log \
    nnet-train-transitions $dir/$y.mod.mdl "ark:gunzip -c $alidir/ali.*.gz|" $dir/$y.mod.mdl \
    || exit 1;

}
function train_one_iter {

  # Set off jobs doing some diagnostics, in the background.
  $cmd $dir/log/compute_prob_valid.$y.log \
    nnet-compute-prob $dir/$y.mdl ark:$dir/valid_diagnostic.egs &
  $cmd $dir/log/compute_prob_train.$y.log \
    nnet-compute-prob $dir/$y.mdl ark:$dir/train_diagnostic.egs &

  echo "Training neural net (pass $y)"
  if [ -f $dir/$y.mod.mdl ]; then
    if [ $dir/$y.mdl -nt $dir/$y.mod.mdl ]; then
      echo "Error: $dir/$y.mdl is newer than $dir/$y.mod.mdl, maybe you need to clean up and rerun?"
      exit 1;
    fi
    mdl=$dir/$y.mod.mdl # In case we made some modification to the model,
      # such as adding a hidden layer.
  else
    mdl=$dir/$y.mdl 
  fi

  $cmd $parallel_opts JOB=1:$num_jobs_nnet $dir/log/train.$y.JOB.log \
    nnet-shuffle-egs --buffer-size=$shuffle_buffer_size --srand=$y \
    ark:$dir/egs/egs.JOB.$[$y%$iters_per_epoch].ark ark:- \| \
    nnet-train-parallel --num-threads=$num_threads --minibatch-size=$minibatch_size \
    --srand=$y $mdl ark:- $dir/$[$y+1].JOB.mdl \
       || exit 1;

  nnets_list=
  for n in `seq 1 $num_jobs_nnet`; do
    nnets_list="$nnets_list $dir/$[$y+1].$n.mdl"
  done
  
  $cmd $dir/log/average.$y.log \
    nnet-am-average $nnets_list $dir/$[$y+1].mdl || exit 1;

  rm $dir/$y.mod.mdl $nnets_list 2>/dev/null
  return 0;
}


function modify_model_if_needed () {

  # If needed, add hidden layers.  E.g. if add_layers_period=3 and num_hidden_layers_per_eon=3, 
  # mix up on iters  3, 6 (this would give us 3 hidden layers as we start with one).

  tmp=$[$add_layers_period*$num_hidden_layers_per_eon]
  if [ $tmp -ge $[num_iters_per_eon-1] ]; then
    echo "Error: not enough iterations per eon to add layers and mix up, $tmp vs $num_iters_per_eon"
    echo "Try increasing --num-epochs or decreasing --samples-per-iter"
    exit 1;
  fi

  if [ $[$x % $add_layers_period ] -eq 0 ] && [ $x -gt 0 ]; then
    n=$[$x/$add_layers_period] # n = 1, 2 ..
    if [ $n -lt $num_hidden_layers_per_eon ]; then # e.g. n = 1, 2.
      echo "Adding new hidden layer"
      # Add a normal hidden layer with ReLU nonlinearity.  We don't randomize this, we randomize
      # the layer that goes to the softmax layer (nnet-insert does this by default).
      param_stddev=`perl -e "print 1.0/sqrt($hidden_layer_size);"` || exit 1
      learning_rate=`perl -e '($x,$n,$i,$f)=@ARGV; print ($x >= $n ? $f : $i*exp($x*log($f/$i)/$n));' $[$x+1] $num_iters_per_eon $initial_learning_rate $final_learning_rate` || exit 1;
      cat <<EOF | tee $dir/nnet.config.$y | nnet-init --srand=$y - - | nnet-insert $dir/$y.mdl - $dir/$y.mod.mdl || exit 1
AffineComponentPreconditioned input-dim=$hidden_layer_size output-dim=$hidden_layer_size alpha=$alpha max-change=$max_change learning-rate=$learning_rate param-stddev=$param_stddev bias-stddev=2
RectifiedLinearComponent dim=$hidden_layer_size
EOF
    fi
    if [ $n -eq $num_hidden_layers_per_eon ]; then # e.g. n = 3
      if [ $[$eon+1] -eq $num_eons ]; then # last eon: mix-up, if applicable
        if [ $mix_up -gt $num_pdfs ]; then
          $cmd $dir/log/mix_up.$y.log \
            nnet-am-mixup --min-count=10 --num-mixtures=$mix_up \
            $dir/$y.mdl $dir/$y.mod.mdl || exit 1;
          mixed_up=true
          echo "Mixed up from $num_pdfs to $mix_up"
        else
          echo "Not mixing up because mix-up=$mix_up, vs num-pdfs=$num_pdfs"
        fi
      fi
    fi
  fi

  if [ $eon -gt 0 ] && [ $x -eq 0 ]; then
    do_eon_start_computation;
  fi
}

function modify_learning_rates() {
  # Modify the learning rates of the trainable layers in the model.  For
  # the layers from previous eons, leave them at the final learning rate,
  # but for the layers added in the current eon, use the current learning
  # learning rate from an exponentially decreasing schedule.
  learning_rate=`perl -e '($x,$n,$i,$f)=@ARGV; print ($x >= $n ? $f : $i*exp($x*log($f/$i)/$n));' $[$x+1] $num_iters_per_eon $initial_learning_rate $final_learning_rate`;
  
  ! num_updatable_layers=`nnet-am-info $dir/$[$y+1].mdl | grep learning-rate | wc -l` 2>/dev/null \
     && echo "Error getting info from $dir/$[$y+1].mdl" && exit 1;

  # The number of layers that require a fixed learning rate is the number of
  # previous eons ($eon) times (the number of hidden layers per eon + 1).
  # It's + 1 because for each previous eon, we still have the matrix that was derived
  # from the output layer, that goes to size $pre_splice_dim -- this is updatable.
  num_fixed_layers=$[$eon*($num_hidden_layers_per_eon+1)];
  # for the first num_hidden_layers_per_eon, use $final_learning_rate, else use
  # $learning_rate.

  learning_rates=`perl -we '($nl,$nf,$lr,$flr) = @ARGV; for ($n=0; $n<$nl;$n++) { push @A,  ($n < $nf ? $flr : $lr); } 
      print join(":", @A);' $num_updatable_layers $num_fixed_layers $learning_rate $old_layer_learning_rate`

  nnet-am-copy --learning-rates=$learning_rates $dir/$[$y+1].mdl $dir/$[$y+1].mdl 2>$dir/log/learning_rate.$y.log
  
}

y=0 # y is the iteration counter that is used to number models.
eon=0 # this is the eon counter.
mixed_up=false

while [ $eon -lt $num_eons ]; do
  x=0 # x is the iteration counter within the eon.
  while [ $x -lt $num_iters_per_eon ]; do
    if [ $stage -le $y ]; then

      modify_model_if_needed || exit 1;

      train_one_iter || exit 1;

      rm $dir/$y.mod.mdl 2>/dev/null

      modify_learning_rates || exit 1;

    fi
    y=$[$y+1]
    x=$[$x+1]
  done
  eon=$[$eon+1]
done

if $realign; then
  if [ $stage -le $y ]; then
    echo "Realigning data (pass $y)"
    $cmd JOB=1:$nj $dir/log/align.$y.JOB.log \
      nnet-align-compiled $scale_opts --beam=$beam --retry-beam=$retry_beam $dir/$y.mdl \
      "ark:gunzip -c $dir/fsts.JOB.gz|" "$feats" \
      "ark:|gzip -c >$dir/ali.JOB.gz" || exit 1;
  fi
fi

x=0
while [ $x -lt $num_iters_extra ]; do
  if [ $stage -le $y ]; then  
    train_one_iter || exit 1;
  fi
  y=$[$y+1]
  x=$[$x+1]
done


if [ $num_iters_combine -gt $num_iters_extra ]; then
  echo "Number of iterations for combination --num-iters-combine will be limited"
  echo "to the number of iterations with constant learning rate, i.e. $num_iters_extra"
  num_iters_combine=$num_iters_extra
fi

first_combine_iter=$[$y-$num_iters_combine]
z=$first_combine_iter;
nnets_to_combine=
while [ $z -le $y ]; do
  nnets_to_combine="$nnets_to_combine $dir/$z.mdl"
  z=$[$z+1]
done

if [ $stage -le $y ]; then
  echo "Doing final combination of model"
  # mb is the minibatch size... we work out an efficient value to use.
  mb=$[($num_valid_frames_combine+$num_train_frames_combine+$num_threads-1)/$num_threads]
  $cmd $parallel_opts $dir/log/combine.log \
    nnet-combine-fast --num-threads=$num_threads --verbose=3 --minibatch-size=$mb \
     $nnets_to_combine ark:$dir/combine.egs $dir/final.mdl || exit 1;
fi


if $cleanup; then
  echo Cleaning up data
  echo Removing training examples
  rm -r $dir/egs
  echo Removing most of the models
  for x in `seq 0 $[$y-1]`; do
    rm $dir/$x.mdl $dir/$x.mod.mdl 2>/dev/null
  done
  rm $dir/raw*.net $dir/temp.mdl 2>/dev/null
fi

$cmd $dir/log/compute_prob_valid.final.log \
  nnet-compute-prob $dir/final.mdl ark:$dir/valid_diagnostic.egs &
$cmd $dir/log/compute_prob_train.final.log \
  nnet-compute-prob $dir/final.mdl ark:$dir/train_diagnostic.egs &

echo Done
