#!/bin/bash

# Copyright 2012  Johns Hopkins University (Author: Daniel Povey).  Apache 2.0.

# Neural net training on top of conventional features-- for best results
# use LDA+MLLT+SAT features, but it helps to have a higher dimension than
# usual (e.g. 50 or 60, versus the more typical 40).  [ the feature dim
# can be set via the --dim option to the script train_lda_mllt.sh ].
# This is a relatively simple neural net training setup that doesn't
# use a two-level tree or any mixture-like stuff.


# Begin configuration section.
cmd=run.pl
num_iters=5   # Total number of iterations
num_valid_utts=300 # held-out utterances.
num_valid_frames=5000 # a subset of the frames in "valid_utts".
minibatch_size=1000
minibatches_per_phase=100
samples_per_iteration=200000 # each iteration of training, see this many samples
                             # per job.
num_jobs_nnet=4 # Number of neural net jobs to run in parallel.
start_parallel=1 # First iteration (in zero-based numbering) to actually do in parallel.
num_hidden_layers=2
initial_num_hidden_layers=1  # we'll add the rest one by one.
num_parameters=2000000 # 2 million parameters by default.
stage=-5
realign_iters=""
beam=10
retry_beam=40
scale_opts="--transition-scale=1.0 --acoustic-scale=0.1 --self-loop-scale=0.1"
parallel_opts=
nnet_config_opts=
splice_width=4 # meaning +- 4 frames on each side for second LDA
lda_dim=250
randprune=4.0 # speeds up LDA.
# If you specify alpha, then we'll do the "preconditioned" update.
alpha=
shrink=false # only applies to iterations before "start_parallel".
# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;


if [ $# != 4 ]; then
  echo "Usage: steps/train_nnet_cpu_parallel.sh <data> <lang> <ali-dir> <exp-dir>"
  echo " e.g.: steps/train_nnet_cpu_parallel.sh data/train_si84 data/lang \\"
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
      --lda-mat $splice_width $lda_dim $dir/lda.mat \
      --initial-num-hidden-layers $initial_num_hidden_layers $dir/hidden_layer.config \
      $feat_dim $num_leaves $num_hidden_layers $num_parameters \
      > $dir/nnet.config || exit 1;
  else
    utils/nnet-cpu/make_nnet_config.pl $nnet_config_opts \
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
  echo "Creating subset of frames of validation set."
  $cmd $dir/log/create_valid_subset.log \
    nnet-randomize-frames $nnet_context_opts --num-samples=$num_valid_frames --srand=0 \
       "$valid_feats" "ark,cs:gunzip -c $dir/ali.*.gz | ali-to-pdf $dir/0.mdl ark:- ark:- |" \
     ark:$dir/valid.egs || exit 1;
fi



x=0
while [ $x -lt $num_iters ]; do
  # note: archive for aligments won't be sorted as the shell glob "*" expands
  # them in alphabetic not numeric order, so we can't use ark,s,cs: below, only
  # ark,cs which means the features are in sorted order [hence alignments will
  # be called in sorted order (cs).
  if [ $stage -le $x ]; then
    if echo $realign_iters | grep -w $x >/dev/null; then
      echo "Realigning data (pass $x)"
      $cmd JOB=1:$nj $dir/log/align.$x.JOB.log \
        nnet-align-compiled $scale_opts --beam=$beam --retry-beam=$retry_beam "$dir/$x.mdl" \
         "ark:gunzip -c $dir/fsts.JOB.gz|" "$split_feats" \
        "ark:|gzip -c >$dir/ali.JOB.gz" || exit 1;
    fi
    echo "Training neural net (pass $x)"
    if [ $x -gt 0 ] && [ $x -le $[$num_hidden_layers-$initial_num_hidden_layers] ]; then
      mdl="nnet-init --srand=$x $dir/hidden_layer.config - | nnet-insert $dir/$x.mdl - - |"
    else
      mdl=$dir/$x.mdl
    fi
    m=$minibatches_per_phase

    if [ $x -lt $start_parallel ]; then # Not parallel; just train in the standard way.
      $cmd $parallel_opts $dir/log/train.$x.log \
        nnet-randomize-frames $nnet_context_opts --num-samples=$samples_per_iteration \
        --srand=$x "$feats" \
        "ark,cs:gunzip -c $dir/ali.*.gz | ali-to-pdf $dir/$x.mdl ark:- ark:- |" ark:- \| \
        nnet-train-simple --minibatch-size=$minibatch_size --minibatches-per-phase=$m \
          --verbose=2 "$mdl" ark:- $dir/$[$x+1].mdl \
        || exit 1;
      if $shrink; then
        $cmd $parallel_opts $dir/log/shrink.$x.log \
          nnet-shrink $dir/$[$x+1].mdl ark:$dir/valid.egs $dir/$[$x+1].mdl \
          || exit 1;
      fi
    else
      egs_list=
      nnets_list=
      for n in `seq 1 $num_jobs_nnet`; do
         egs_list="$egs_list ark:$dir/egs.tmp.$n"
         nnets_list="$nnets_list $dir/$[$x+1].$n.mdl"
      done
      $cmd $parallel_opts $dir/log/randomize.$x.log \
        nnet-randomize-frames $nnet_context_opts --num-samples=$[$samples_per_iteration*$num_jobs_nnet] \
        --srand=$x "$feats" \
        "ark,cs:gunzip -c $dir/ali.*.gz | ali-to-pdf $dir/$x.mdl ark:- ark:- |" ark:- \| \
         nnet-copy-egs ark:- $egs_list || exit 1;
      $cmd $parallel_opts JOB=1:$num_jobs_nnet $dir/log/train.$x.JOB.log \
        nnet-train-simple \
          --minibatch-size=$minibatch_size --minibatches-per-phase=$m \
          --verbose=2 "$mdl" ark:$dir/egs.tmp.JOB $dir/$[$x+1].JOB.mdl \
         || exit 1;
      $cmd $parallel_opts $dir/log/combine.$x.log \
         nnet-combine "$mdl" $nnets_list ark:$dir/valid.egs $dir/$[$x+1].mdl.tmp || exit 1;

      # Use information from the logging output of nnet-combine to update the learning rates.
      utils/nnet-cpu/update_learning_rates.pl $dir/log/combine.$x.log $dir/$[$x+1].mdl.tmp $dir/$[$x+1].mdl \
        2>$dir/log/update_learning_rate.$x.log || exit 1;
       
      rm $dir/$[$x+1].mdl.tmp $nnets_list
    fi
  fi
  x=$[$x+1]
done

rm $dir/final.mdl 2>/dev/null
ln -s $x.mdl $dir/final.mdl

echo Done
