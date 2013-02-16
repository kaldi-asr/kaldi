#!/bin/bash

# Copyright 2012  Johns Hopkins University (Author: Daniel Povey).  Apache 2.0.

# MMI (or boosted MMI) training (A.K.A. sequence training) of a neural net based 
# system as trained by train_nnet_cpu.sh


# Begin configuration section.
cmd=run.pl
epochs_per_ebw_iter=1 # Number of times we iterate over the whole
                       # data each time we do an "EBW" iteration.
num_ebw_iters=4 # Number of "EBW" iterations.
initial_learning_rate=0.001 # learning rate we start with.
learning_rate_factor=1.0 # factor by which we change the learning
                         # rate each iteration (should be <= 1.0)
E=2.0  # this is slightly analogous to the constant E used in
       # Extended Baum-Welch updates of GMMs.  It slows down (and
       # somewhat regularizes) the update.

minibatch_size=256 # since the learning rate is always quite low compared with
                   # what we have at the start of ML training, we can probably
                   # afford a somewhat higher minibatch size than there, as
                   # there is less risk of instability.

samples_per_iter=400000 # each phase of training, see this many samples
                         # per job.  Note: this is a kind of suggestion; we
                         # will actually find a number that will make the
                          # #iters per epoch a whole number.
num_jobs_nnet=8 # Number of neural net training jobs to run in parallel.
                # not the same as the num-jobs (nj) which will be the same as the
                # alignment and denlat directories.
stage=0
sub_stage=-3 # this can be used to start from a particular sub-iteration of an
             # iteration
acwt=0.1
boost=0.0  # boosting for BMMI (you can try 0.1).. this is applied per frame.
transform_dir=  # Note: by default any transforms in $alidir will be used.

parallel_opts="-pe smp 16" # by default we use 16 threads; this lets the queue know.
io_opts="-tc 10" # max 5 jobs running at one time (a lot of I/O.)
num_threads=16 # number of threads for neural net trainer..
mkl_num_threads=1
random_copy=false
# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;


if [ $# != 6 ]; then
  echo "Usage: steps/train_nnet_cpu_mmi.sh [opts] <data> <lang> <src-dir> <ali-dir> <denlat-dir> <exp-dir>"
  echo ""
  echo "Main options (for others, see top of script file)"
  echo "Note, the terminology is: each iteration of EBW we do multiple epochs; each epoch"
  echo " we have multiple iterations of training (note the same as the EBW iters)."
  echo "  --config <config-file>                           # config file containing options"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --num-ebw-iters <#iters|4>                       # number of pseudo-Extended-Baum-Welch iterations (default: 4)"
  echo "  --epochs-per-ebw-iter <#epochs|1>                # number of times to see all the data per EBW iter."
  echo "  --initial-learning-rate <initial-lrate|0.005>    # learning rate to use on the first iteration"
  echo "  --learning-rate-factor <lrate-factor|1.0>        # Factor by which to change the learning rate on each"
  echo "                                                   # EBW iteration (should be <= 1.0)"
  echo "  --num-jobs-nnet <num-jobs|8>                     # Number of parallel jobs to use for main neural net"
  echo "                                                   # training (will affect results as well as speed; try 8, 16)."
  echo "                                                   # Note: if you increase this, you may want to also increase"
  echo "                                                   # the learning rate."
  echo "  --num-threads <num-threads|16>                   # Number of parallel threads per job (will affect results"
  echo "                                                   # as well as speed; may interact with batch size; if you increase"
  echo "                                                   # this, you may want to decrease the batch size."
  echo "  --parallel-opts <opts|\"-pe smp 16\">            # extra options to pass to e.g. queue.pl for processes that"
  echo "                                                   # use multiple threads."
  echo "  --io-opts <opts|\"-tc 10\">                      # Options given to e.g. queue.pl for any especially I/O intensive jobs"
  echo "  --minibatch-size <minibatch-size|128>            # Size of minibatch to process (note: product with --num-threads"
  echo "                                                   # should not get too large, e.g. >2k)."
  echo "  --samples-per-iter <#samples|400000>             # Number of samples of data to process per iteration, for each"
  echo "                                                   # process.  Note: this will get modified to a number that will"
  echo "                                                   # divide the data into a whole number of pieces."
  echo "  --transform-dir <dir>                            # Directory to find fMLLR transforms; if not specified, "
  echo "                                                   # $alidir will be used if it has transforms"
  echo "  --stage <stage|0>                                # Used to run a partially-completed training process from somewhere in"
  echo "                                                   # the middle."
  echo "  --sub-stage <sub-stage|0>                        # In conjunction with --stage, can be used to start a partially-completed"
  echo "                                                   # training process (refers to the phase number)"
  

  exit 1;
fi

data=$1
lang=$2
srcdir=$3
alidir=$4 # Also used for transforms by default, if transform-dir not specified.
denlatdir=$5
dir=$6 # experimental directory

# Check that some files exist, mostly to verify correct directory arguments.
for f in $data/feats.scp $lang/L.fst $srcdir/final.mdl $srcdir/final.mat $alidir/ali.1.gz $denlatdir/lat.1.gz; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

mkdir -p $dir/log
cp $srcdir/tree $dir
learning_rate=$initial_learning_rate
if [ $stage -ge -1 ]; then
  $cmd $dir/log/copy_initial.log \
     nnet-am-copy --learning-rate=$learning_rate $srcdir/final.mdl $dir/0.1.mdl
fi

nnet_context_opts="--left-context=`nnet-am-info $dir/0.1.mdl 2>/dev/null | grep -w left-context | awk '{print $2}'` --right-context=`nnet-am-info $dir/0.1.mdl 2>/dev/null | grep -w right-context | awk '{print $2}'`" || exit 1;

silphonelist=`cat $lang/phones/silence.csl` || exit 1;

nj=`cat $alidir/num_jobs` || exit 1;  # number of jobs in alignment dir...
nj2=`cat $denlatdir/num_jobs` || exit 1; # number of jobs in denlat dir
[ "$nj" != "$nj2" ] && echo "Mismatch in #jobs $nj vs $nj2" && exit 1;

sdata=$data/split$nj

splice_opts=`cat $alidir/splice_opts 2>/dev/null`
cp $alidir/splice_opts $dir 2>/dev/null
cp $alidir/final.mat $dir || exit 1;
cp $alidir/tree $dir

all_feats="ark,s,cs:apply-cmvn --norm-vars=false --utt2spk=ark:$data/utt2spk scp:$data/cmvn.scp scp:$data/feats.scp ark:- | splice-feats $splice_opts ark:- ark:- | transform-feats $dir/final.mat ark:- ark:- |"
feats="ark,s,cs:apply-cmvn --norm-vars=false --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | splice-feats $splice_opts ark:- ark:- | transform-feats $dir/final.mat ark:- ark:- |"

if [ -z "$transform_dir" ] && [ -f "$alidir/trans.1" ]; then 
  # --transform-dir option not set and $alidir has transforms in it.
  transform_dir=$alidir
fi

if [ -f $alidir/trans.1 ]; then
  echo "$0: using transforms from $alidir"
  all_feats="$all_feats transform-feats --utt2spk=ark:$data/utt2spk 'ark:cat $alidir/trans.*|' ark:- ark:- |"
  feats="$feats transform-feats --utt2spk=ark:$sdata/JOB/utt2spk ark:$alidir/trans.JOB ark:- ark:- |"
else
  echo "$0: not using fMLLR transforms (assuming unadapted system)"
fi

echo "$0: working out number of frames of training data"

num_frames=`feat-to-len scp:$data/feats.scp ark,t:- | awk '{x += $2;} END{print x;}'` || exit 1;

# round to closest int
iters_per_epoch=`perl -e "print int($num_frames/($samples_per_iter * $num_jobs_nnet) + 0.5);"` || exit 1;
[ $iters_per_epoch -eq 0 ] && iters_per_epoch=1
samples_per_iter_real=$[$num_frames/($num_jobs_nnet*$iters_per_epoch)]

echo "Every EBW iteration, splitting the data up into $iters_per_epoch iterations,"
echo "giving samples-per-iteration of $samples_per_iter_real (you requested $samples_per_iter)."

mkdir -p $dir/post $dir/egs

num_epochs=$[$num_ebw_iters*$epochs_per_ebw_iter]

x=0
while [ $x -lt $num_epochs ]; do
  z=$[$x / $epochs_per_ebw_iter];  # z is the (generally) smaller iteration number that identifies the EBW pass.
  if [ $x -eq $[$z * $epochs_per_ebw_iter] ]; then
    first_iter_of_epoch=true
    echo "Starting pass $z of EBW"
  else
    first_iter_of_epoch=false
  fi
  echo "Epoch $x of $num_epochs"

  if [ $stage -le $x ] && $first_iter_of_epoch; then
    if [ $stage -lt $x ] || [ $sub_stage -le -3 ]; then
      # First get the per-frame posteriors, by rescoring the lattices; this
      # process also gives us at the same time the posteriors of each state for
      # each frame (by default, pruned to 0.01 with a randomized algorithm).
      # The matrix-logprob stage produces a diagnostic and passes the pseudo-log-like
      # matrix through unchanged.  (Note: nnet-logprob2-parallel can use up to
      # $num_threads threads, but in practice it may be limited by the speed of
      # the other elements of the pipe.
      $cmd $parallel_opts JOB=1:$nj $dir/log/post.$z.JOB.log \
        nnet-logprob2-parallel --num-threads=$num_threads $dir/$x.1.mdl "$feats" \
          "ark:|prob-to-post ark:- ark:- | gzip -c >$dir/post/smooth_post.$z.JOB.gz" ark:- \| \
        matrix-logprob ark:- "ark:gunzip -c $alidir/ali.JOB.gz | ali-to-pdf $dir/$x.1.mdl ark:- ark:-|" ark:- \| \
        lattice-rescore-mapped $dir/$x.1.mdl "ark:gunzip -c $denlatdir/lat.JOB.gz|" ark:- ark:- \| \
        lattice-boost-ali --b=$boost --silence-phones=$silphonelist $dir/$x.1.mdl ark:- "ark:gunzip -c $alidir/ali.JOB.gz|" ark:- \| \
        lattice-to-post --acoustic-scale=$acwt ark:- ark:- \| \
        post-to-pdf-post $dir/$x.1.mdl ark:- "ark:|gzip -c >$dir/post/den_post.$z.JOB.gz" || exit 1;
    fi
    if [ $stage -lt $x ] || [ $sub_stage -le -2 ]; then
      # run nnet-get-egs for all files, to get the training examples for each frame--
      # combines the feature and label/posterior information.  The posterior information
      # consists of 2 things: the numerator posteriors from the alignments, the denominator
      # posteriors from the lattices (times -1), and the smoothing posteriors from the 
      # neural net log-probs (times E).  
      # We copy the examples for each job round-robin to multiple archives, one for each
      # of 1...$num_jobs_nnet.  
      egs_out=""
      for n in `seq 1 $num_jobs_nnet`; do
        # indexes are egs_orig.$z.$num_jobs_nnet.$nj
        egs_out="$egs_out ark:$dir/egs/egs_orig.$z.$n.JOB.ark"
      done
      $cmd JOB=1:$nj $dir/log/get_egs.$z.JOB.log \
         ali-to-pdf $dir/$x.1.mdl "ark:gunzip -c $alidir/ali.JOB.gz|" ark:- \| \
         ali-to-post ark:- ark:- \| \
         sum-post --scale2=$E ark:- "ark:gunzip -c $dir/post/smooth_post.$z.JOB.gz|" ark:- \| \
         sum-post --scale2=-1.0 ark:- "ark:gunzip -c $dir/post/den_post.$z.JOB.gz|" ark:- \| \
         nnet-get-egs $nnet_context_opts "$feats" ark:- ark:- \| \
         nnet-copy-egs ark:- $egs_out || exit 1;
      rm $dir/post/smooth_post.$z.*.gz $dir/post/den_post.$z.*.gz 
    fi
    if $first_iter_of_epoch; then
      # Diagnostics-- work out an extra term in the objf that we have to add to
      # what we get from the nnet training.
      tail -n 50 $dir/log/post.$z.*.log | perl -e '$acwt=shift @ARGV; $acwt>0.0 || die "bad acwt"; while(<STDIN>) { if (m|lattice-to-post.+Overall average log-like/frame is (\S+) over (\S+) frames.  Average acoustic like/frame is (\S+)|) { $tot_den_lat_like += $1*$2; $tot_frames += $2; } if (m|matrix-logprob.+Average log-prob per frame is (\S+) over (\S+) frames|) { $tot_num_like += $1*$2; $tot_num_frames += $2; } } if (abs($tot_frames - $tot_num_frames) > 0.01*($tot_frames + $tot_num_frames)) { print STDERR "#frames differ $tot_frames vs $tot_num_frames\n"; }  $tot_den_lat_like /= $tot_frames; $tot_num_like /= $tot_num_frames; $objf = $acwt * $tot_num_like - $tot_den_lat_like; print $objf."\n"; ' $acwt > $dir/log/objf.$z.log
      echo "Objf on EBW iter $z is `cat $dir/log/objf.$z.log`"
    fi
    if [ $stage -lt $x ] || [ $sub_stage -le -1 ]; then
      echo "Merging training examples across original #jobs ($nj), and "
      echo "splitting across number of nnet jobs $num_jobs_nnet"
      egs_out2=""
      for n in `seq 1 $iters_per_epoch`; do
        # indexes of egs_merged are: egs_merged.$z.$iters_per_epoch.$num_jobs_nnet
        egs_out2="$egs_out2 ark:$dir/egs/egs_merged.$z.$n.JOB.ark"
      done
      # Note: in the following command, JOB goes from 1 to $num_jobs_nnet, so one
      # job per parallel training job (different from the previous command).
      # We sum up over the index JOB in the previous $cmd, and write to multiple
      # archives, this time one for each "sub-iter".
      # indexes of egs_orig are: egs_orig.$z.$num_jobs_nnet.$nj
      $cmd $io_opts JOB=1:$num_jobs_nnet $dir/log/merge_and_split.$x.JOB.log \
        cat $dir/egs/egs_orig.$z.JOB.*.ark \| \
        nnet-copy-egs --random=$random_copy "--srand=\$[JOB+($x*$num_jobs_nnet)]" \
          ark:- $egs_out2 '&&' rm $dir/egs/egs_orig.$z.JOB.*.ark || exit 1;
    fi
    if [ $stage -lt $x ] || [ $sub_stage -le 0 ]; then
      echo "Randomizing order of examples in each job"
      for n in `seq 1 $iters_per_epoch`; do
        s=$[$num_jobs_nnet*($n+($iters_per_epoch*$z))] # for srand
        $cmd $io_opts JOB=1:$num_jobs_nnet $dir/log/shuffle.$z.$n.JOB.log \
          nnet-shuffle-egs "--srand=\$[JOB+$s]" \
          ark:$dir/egs/egs_merged.$z.$n.JOB.ark ark:$dir/egs/egs.$z.$n.JOB.ark '&&' \
          rm $dir/egs/egs_merged.$z.$n.JOB.ark || exit 1;
      done
    fi
  fi
  if [ $stage -le $x ]; then
    # This block does the $iters_per_epoch iters of training.
    y=1; # y is the "sub-iteration" number.
    while [ $y -le $iters_per_epoch ]; do
      echo "Iteration $x, sub-iteration $y"
      if [ $stage -lt $x ] || [ $sub_stage -le $y ]; then
        $cmd $parallel_opts JOB=1:$num_jobs_nnet $dir/log/train.$x.$y.JOB.log \
          nnet-train-parallel --num-threads=$num_threads --minibatch-size=$minibatch_size \
          $dir/$x.$y.mdl ark:$dir/egs/egs.$z.$y.JOB.ark $dir/$x.$y.JOB.mdl \
          || exit 1;
        nnets_list=
        for n in `seq 1 $num_jobs_nnet`; do
          nnets_list="$nnets_list $dir/$x.$y.$n.mdl"
        done
        if [ $y -eq $iters_per_epoch ]; then next_mdl=$dir/$[$x+1].1.mdl
        else next_mdl=$dir/$x.$[$y+1].mdl; fi
        # Average the parameters of all the parallel jobs.
        $cmd $dir/log/average.$x.$y.log \
           nnet-am-average $nnets_list $next_mdl || exit 1;
        rm $nnets_list
      fi
      y=$[$y+1]
    done
  fi
  if [ $learning_rate_factor != 1.0 ]; then
    learning_rate=`perl -e "print $learning_rate * $learning_rate_factor;"`;
    ! nnet-am-copy --print-args=false --learning-rate=$learning_rate $dir/$[$x+1].1.mdl $dir/$[$x+1].1.mdl && \
       echo Error changing learning rate of neural net && exit 1;
  fi
  x=$[$x+1]
done

rm $dir/final.mdl 2>/dev/null
ln -s $x.1.mdl $dir/final.mdl

echo Done
