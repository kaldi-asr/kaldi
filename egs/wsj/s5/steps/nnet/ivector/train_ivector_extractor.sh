#!/usr/bin/env bash

# Copyright   2013  Daniel Povey
#             2016  Brno University of Technology (Author: Karel Vesely)
# Apache 2.0.

# This script is modified from ^/egs/sre08/v1/sid/train_ivector_extractor.sh.
# It trains an iVector extractor for use in DNN training.

# This script trains the i-vector extractor.  Note: there are 3 separate levels
# of parallelization: num_threads, num_processes, and num_jobs.  This may seem a
# bit excessive.  It has to do with minimizing memory usage and disk I/O,
# subject to various constraints.  The "num_threads" is how many threads a
# program uses; the "num_processes" is the number of separate processes a single
# job spawns, and then sums the accumulators in memory.  Our recommendation:
#  - Set num_threads to the minimum of (4, or how many virtual cores your machine has).
#    (because of needing to lock various global quantities, the program can't
#    use many more than 4 threads with good CPU utilization).
#  - Set num_processes to the number of virtual cores on each machine you have, divided by
#    num_threads.  E.g. 4, if you have 16 virtual cores.   If you're on a shared queue
#    that's busy with other people's jobs, it may be wise to set it to rather less
#    than this maximum though, or your jobs won't get scheduled.  And if memory is
#    tight you need to be careful; in our normal setup, each process uses about 5G.
#  - Set num_jobs to as many of the jobs (each using $num_threads * $num_processes CPUs)
#    your queue will let you run at one time, but don't go much more than 10 or 20, or
#    summing the accumulators will possibly get slow.  If you have a lot of data, you
#    may want more jobs, though.

# Begin configuration section.
nj=10   # this is the number of separate queue jobs we run, but each one
        # contains num_processes sub-jobs.. the real number of threads we
        # run is nj * num_processes * num_threads, and the number of
        # separate pieces of data is nj * num_processes.
num_threads=4
num_processes=2 # each job runs this many processes, each with --num-threads threads
cmd="run.pl"
stage=-4
ivector_dim=100 # dimension of the extracted i-vector
num_iters=10
num_gselect=5 # Gaussian-selection using diagonal model: number of Gaussians to select
posterior_scale=0.1 # Scale on the acoustic posteriors, intended to account for
                    # inter-frame correlations.
min_post=0.025 # Minimum posterior to use (posteriors below this are pruned out)
               # caution: you should use the same value in the online-estimation
               # code.
subsample=2  # This speeds up the training: training on every 2nd feature
             # (configurable) Since the features are highly correlated across
             # frames, we don't expect to lose too much from this.
parallel_opts=  # ignored now.
cleanup=true
# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;


if [ $# != 3 ]; then
  echo "Usage: $0 <data> <diagonal-ubm-dir> <extractor-dir>"
  echo " e.g.: $0 data/train exp/nnet2_online/diag_ubm/ exp/nnet2_online/extractor"
  echo "main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config containing options"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --num-iters <#iters|10>                          # Number of iterations of E-M"
  echo "  --nj <n|10>                                      # Number of jobs (also see num-processes and num-threads)"
  echo "  --num-processes <n|4>                            # Number of processes for each queue job (relates"
  echo "                                                   # to summing accs in memory)"
  echo "  --num-threads <n|4>                              # Number of threads for each process (can't be usefully"
  echo "                                                   # increased much above 4)"
  echo "  --stage <stage|-4>                               # To control partial reruns"
  echo "  --num-gselect <n|5>                              # Number of Gaussians to select using"
  echo "                                                   # diagonal model."
  exit 1;
fi

set -euxo pipefail

data=$1
srcdir=$2
dir=$3

for f in $srcdir/final.dubm $data/feats.scp; do
  [ ! -f $f ] && echo "No such file $f" && exit 1;
done

# Set various variables.
mkdir -p $dir/log
nj_full=$[$nj*$num_processes]
sdata=$data/split$nj_full;
utils/split_data.sh $data $nj_full

cp $srcdir/final.dubm $dir

## Set up features.
gmm_feats="ark,s,cs:copy-feats scp:$sdata/JOB/feats.scp ark:- | subsample-feats --n=$subsample ark:- ark:- |"
feats="$gmm_feats"

# Initialize the i-vector extractor using the input GMM, which is converted to
# full because that's what the i-vector extractor expects.  Note: we have to do
# --use-weights=false to disable regression of the log weights on the ivector,
# because that would make the online estimation of the ivector difficult (since
# the online/real-time ivector estimation is the whole point of this script).
if [ $stage -le -2 ]; then
  $cmd $dir/log/init.log \
    ivector-extractor-init --ivector-dim=$ivector_dim --use-weights=false \
     "gmm-global-to-fgmm $dir/final.dubm -|" $dir/0.ie
fi

# Do Gaussian selection and posterior extracion

# if we subsample frame, modify the posterior-scale; this is likely
# to make the original posterior-scale (before subsampling) suitable.
modified_posterior_scale=$(perl -e "print $posterior_scale * $subsample;");

if [ $stage -le -1 ]; then
  echo $nj_full > $dir/num_jobs
  echo "$0: doing Gaussian selection and posterior computation"
  $cmd JOB=1:$nj_full $dir/log/post.JOB.log \
    gmm-global-get-post --n=$num_gselect --min-post=$min_post $dir/final.dubm "$gmm_feats" ark:- \| \
    scale-post ark:- $modified_posterior_scale "ark:|gzip -c >$dir/post.JOB.gz"
else
  # make sure we at least have the right number of post.*.gz files.
  if ! [ $nj_full -eq $(cat $dir/num_jobs) ]; then
    echo "Num-jobs mismatch $nj_full versus $(cat $dir/num_jobs)"
    exit 1
  fi
fi

x=0
while [ $x -lt $num_iters ]; do
  if [ $stage -le $x ]; then
    rm $dir/.error 2>/dev/null || true

    Args=() # bash array of training commands for 1:nj, that put accs to stdout.
    for j in $(seq $nj_full); do
      Args[$j]=`echo "ivector-extractor-acc-stats --num-threads=$num_threads $dir/$x.ie '$feats' 'ark,s,cs:gunzip -c $dir/post.JOB.gz|' -|" | sed s/JOB/$j/g`
    done

    echo "Accumulating stats (pass $x)"
    for g in $(seq $nj); do
      start=$[$num_processes*($g-1)+1]
      $cmd --num-threads $[$num_threads*$num_processes] $dir/log/acc.$x.$g.log \
        ivector-extractor-sum-accs --parallel=true "${Args[@]:$start:$num_processes}" \
          $dir/acc.$x.$g || touch $dir/.error &
    done
    wait
    [ -f $dir/.error ] && echo "Error accumulating stats on iteration $x" && exit 1;

    accs=""
    for j in $(seq $nj); do
      accs+="$dir/acc.$x.$j "
    done
    echo "Summing accs (pass $x)"
    $cmd $dir/log/sum_acc.$x.log \
      ivector-extractor-sum-accs $accs $dir/acc.$x

    echo "Updating model (pass $x)"
    nt=$[$num_threads*$num_processes] # use the same number of threads that
                                      # each accumulation process uses, since we
                                      # can be sure the queue will support this many.
                                      #
                                      # The parallel-opts was either specified by
                                      # the user or we computed it correctly in
                                      # tge previous stages
    $cmd --num-threads $[$num_threads*$num_processes] $dir/log/update.$x.log \
      ivector-extractor-est --num-threads=$nt $dir/$x.ie $dir/acc.$x $dir/$[$x+1].ie
    rm $dir/acc.$x.*

    if $cleanup; then
      rm $dir/acc.$x
      # rm $dir/$x.ie
    fi
  fi
  x=$[$x+1]
done

rm $dir/final.ie 2>/dev/null || true
ln -s $x.ie $dir/final.ie
