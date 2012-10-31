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
num_valid_utts=50
minibatch_size=1000
minibatches_per_phase_it1=50
minibatches_per_phase=200
samples_per_iteration=400000 # each iteration of training, see this many samples.
frequency_power=1.0 # Power used in sampling and corresonding reweighting of stats...
num_hidden_layers=2
initial_num_hidden_layers=1  # we'll add the rest one by one.
num_parameters=2000000 # 2 million parameters by default.
stage=-4
realign_iters=""
beam=10
retry_beam=40
scale_opts="--transition-scale=1.0 --acoustic-scale=0.1 --self-loop-scale=0.1"
parallel_opts=
# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;


if [ $# != 4 ]; then
  echo "Usage: steps/train_nnet_cpu.sh <data> <lang> <ali-dir> <exp-dir>"
  echo " e.g.: steps/train_nnet_cpu.sh data/train_si84 data/lang \\"
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

nj=`cat $alidir/num_jobs` || exit 1;  # number of jobs in alignment dir...
# in this dir we'll have just one job.
sdata=$data/split$nj

mkdir -p $dir/log
echo $nj > $dir/num_jobs
splice_opts=`cat $alidir/splice_opts 2>/dev/null`
cp $alidir/splice_opts $dir 2>/dev/null
cp $alidir/final.mat $dir 2>/dev/null # any LDA matrix...
cp $alidir/tree $dir



# Get list of validation utterances.  sort -R is random sort.
awk '{print $1}' $data/utt2spk | sort -R | head -$num_valid_utts \
    > $dir/valid_uttlist || exit 1;

$cmd $dir/log/convert_valid_alignments.log \
  copy-int-vector "ark,cs:gunzip -c $alidir/ali.*.gz|" ark,t:- \| \
    utils/filter_scp.pl $dir/valid_uttlist \| \
    ali-to-pdf $alidir/final.mdl ark:- "ark,t:|gzip -c >$dir/pdfs.valid.gz" || exit 1;


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
  lda) feats="ark,s,cs:utils/filter_scp.pl --exclude $dir/valid_uttlist $data/feats.scp | apply-cmvn --norm-vars=false --utt2spk=ark:$data/utt2spk scp:$data/cmvn.scp scp:- ark:- | splice-feats $splice_opts ark:- ark:- | transform-feats $alidir/final.mat ark:- ark:- |"
      split_feats="ark,s,cs:apply-cmvn --norm-vars=false --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | splice-feats $splice_opts ark:- ark:- | transform-feats $alidir/final.mat ark:- ark:- |"
      valid_feats="ark,s,cs:utils/filter_scp.pl $dir/valid_uttlist $data/feats.scp | apply-cmvn --norm-vars=false --utt2spk=ark:$data/utt2spk scp:$data/cmvn.scp scp:- ark:- | splice-feats $splice_opts ark:- ark:- | transform-feats $alidir/final.mat ark:- ark:- |"
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


##
if [ $initial_num_hidden_layers -gt $num_hidden_layers ]; then
  echo "Initial num-hidden-layers $initial_num_hidden_layers is greater than final number $num_hidden_layers";
  exit 1;
fi

if [ $stage -le -4 ]; then
  echo "$0: initializing neural net";
  # to hidden.config it will write the part of the config corresponding to a
  # single hidden layer; we need this to add new layers.
  utils/nnet-cpu/make_nnet_config.pl  \
      --initial-num-hidden-layers $initial_num_hidden_layers $dir/hidden_layer.config \
     $feat_dim $num_leaves $num_hidden_layers $num_parameters \
      > $dir/nnet.config || exit 1;
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
    [ $x -eq 0 ] && m=$minibatches_per_phase_it1
    $cmd $parallel_opts $dir/log/train.$x.log \
      nnet-randomize-frames --num-samples=$samples_per_iteration \
      --frequency-power=$frequency_power --srand=$x \
      "$feats" "ark,cs:gunzip -c $dir/ali.*.gz | ali-to-pdf $dir/$x.mdl ark:- ark:- |" ark:- \| \
      nnet-train --minibatch-size=$minibatch_size --minibatches-per-phase=$m \
        --verbose=2 "$mdl" "$valid_feats" "ark,cs:gunzip -c $dir/pdfs.valid.gz|" ark:- $dir/$[$x+1].mdl \
      || exit 1;
  fi
  x=$[$x+1]
done

rm $dir/final.mdl 2>/dev/null
ln -s $x.mdl $dir/final.mdl

echo Done
