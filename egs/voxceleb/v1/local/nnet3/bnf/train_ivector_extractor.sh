#!/bin/bash

# Copyright   2013  Daniel Povey
#             2014  David Snyder
#             2018  Ewald Enzinger
# Apache 2.0.
#
# Modified version of egs/sre08/v1/sid/train_ivector_extractor.sh (commit 26b0746f0d601e60e87615c649d919525cbd8d9d)

# This script trains the i-vector extractor using bottleneck features for posterior
# computation and speaker ID features for data.  Note: there are 3 separate levels
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
num_processes=4 # each job runs this many processes, each with --num-threads threads
cmd="run.pl"
stage=-3
num_gselect=20 # Gaussian-selection using diagonal model: number of Gaussians to select
ivector_dim=400 # dimension of the extracted i-vector
use_weights=false # set to true to turn on the regression of log-weights on the ivector.
num_iters=10
min_post=0.025 # Minimum posterior to use (posteriors below this are pruned out)
num_samples_for_weights=3 # smaller than the default for speed (relates to a sampling method)
cleanup=true
posterior_scale=1.0 # This scale helps to control for successve features being highly
                    # correlated.  E.g. try 0.1 or 0.3
delta_window=3
delta_order=2
add_bnf=false
apply_cmn=true
sum_accs_opt=
# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;


if [ $# != 4 ]; then
  echo "Usage: $0 <fgmm-bnf-model> <data> <data-bnf> <extractor-dir>"
  echo " e.g.: $0 exp/full_ubm_2048/final_bnf.ubm data/train data/train_bnf exp/extractor_bnf"
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
  echo "  --num-gselect <n|20>                             # Number of Gaussians to select using"
  echo "                                                   # diagonal model."
  echo "  --sum-accs-opt <option|''>                       # Option e.g. '-l hostname=a15' to localize"
  echo "                                                   # sum-accs process to nfs server."
  echo "  --delta-window <n|3>                             # number of frames of context used to"
  echo "                                                   # calculate delta"
  echo "  --delta-order <n|2>                              # number of delta features"
  echo "  --add-bnf <true,false|false>                     # if true, append BNF to speaker ID features"
  echo "                                                   # for stats computation"
  echo "  --apply-cmn <true,false|true>                    # if true, apply sliding window cepstral mean"
  echo "                                                   # normalization to features"
  exit 1;
fi

fgmm_bnf_model=$1
data=$2
data_bnf=$3
dir=$4
srcdir=$(dirname $fgmm_bnf_model)

for f in $fgmm_bnf_model $data_bnf/feats.scp $data/feats.scp; do
  [ ! -f $f ] && echo "No such file $f" && exit 1;
done

# Set various variables.
mkdir -p $dir/log
nj_full=$[$nj*$num_processes]
sdata_bnf=$data_bnf/split$nj_full;
utils/split_data.sh $data_bnf $nj_full || exit 1;
sdata=$data/split$nj_full;
utils/split_data.sh $data $nj_full || exit 1;

delta_opts="--delta-window=$delta_window --delta-order=$delta_order"
echo $delta_opts > $dir/delta_opts

parallel_opts="--num-threads $[$num_threads*$num_processes]"
## Set up features.
bnf_feats="ark,s,cs:select-voiced-frames scp:$sdata_bnf/JOB/feats.scp scp:$sdata/JOB/vad.scp ark:- |"

feats="ark,s,cs:add-deltas $delta_opts scp:$sdata/JOB/feats.scp ark:- |"
if $add_bnf; then
  feats="$feats paste-feats ark:- scp:$sdata_bnf/JOB/feats.scp ark:- |"
fi
if $apply_cmn; then
  feats="$feats apply-cmvn-sliding --norm-vars=false --center=true --cmn-window=300 ark:- ark:- |"
fi
feats="$feats select-voiced-frames ark:- scp:$sdata/JOB/vad.scp ark:- |"

if [ $stage -le -3 ]; then
  echo $nj_full > $dir/num_jobs
  cp $fgmm_bnf_model $dir/final_bnf.ubm || exit 1;
  $cmd $dir/log/convert.log \
    fgmm-global-to-gmm $dir/final_bnf.ubm $dir/final_bnf.dubm || exit 1;

  echo "$0: doing Gaussian selection and posterior computation"
  $cmd JOB=1:$nj_full $dir/log/gselect.JOB.log \
    gmm-gselect --n=$num_gselect $dir/final_bnf.dubm "$bnf_feats" ark:- \| \
    fgmm-global-gselect-to-post --min-post=$min_post $dir/final_bnf.ubm "$bnf_feats" ark,s,cs:-  ark:- \| \
    scale-post ark:- $posterior_scale "ark:|gzip -c >$dir/post.JOB.gz" || exit 1;
else
  if ! [ $nj_full -eq $(cat $dir/num_jobs) ]; then
    echo "Num-jobs mismatch $nj_full versus $(cat $dir/num_jobs)"
    exit 1
  fi
fi
num_components=`grep -oP 'number\ of\ gaussians\ \K[0-9]+' <(fgmm-global-info $fgmm_bnf_model 2> /dev/null)`
if [ $stage -le -2 ]; then
  echo "$0: initializing GMM from stats"
  $cmd JOB=1:$nj_full $dir/log/make_fgmm_accs.JOB.log \
    fgmm-global-acc-stats-post "ark,s,cs:gunzip -c $dir/post.JOB.gz|" $num_components \
      "$feats" $dir/fgmm_stats.JOB.acc || exit 1

  $cmd $dir/log/init_fgmm.log \
    fgmm-global-init-from-accs --verbose=2 \
      "fgmm-global-sum-accs - $dir/fgmm_stats.*.acc |" $num_components \
      $dir/final.ubm || exit 1;

  $cleanup && rm $dir/fgmm_stats.*.acc
fi

# Initialize the i-vector extractor using the FGMM input
if [ $stage -le -1 ]; then
  $cmd $dir/log/init.log \
    ivector-extractor-init --ivector-dim=$ivector_dim --use-weights=$use_weights \
      $dir/final.ubm $dir/0.ie || exit 1
fi

x=0
while [ $x -lt $num_iters ]; do
  if [ $stage -le $x ]; then
    rm $dir/.error 2>/dev/null

    Args=() # bash array of training commands for 1:nj, that put accs to stdout.
    for j in $(seq $nj_full); do
      Args[$j]=`echo "ivector-extractor-acc-stats --num-threads=$num_threads --num-samples-for-weights=$num_samples_for_weights $dir/$x.ie '$feats' 'ark,s,cs:gunzip -c $dir/post.JOB.gz|' -|" | sed s/JOB/$j/g`
    done

    echo "Accumulating stats (pass $x)"
    for g in $(seq $nj); do
      start=$[$num_processes*($g-1)+1]
      $cmd $parallel_opts $dir/log/acc.$x.$g.log \
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
    $cmd $sum_accs_opt $dir/log/sum_acc.$x.log \
      ivector-extractor-sum-accs $accs $dir/acc.$x || exit 1;
    echo "Updating model (pass $x)"
    nt=$[$num_threads*$num_processes] # use the same number of threads that
                                      # each accumulation process uses, since we
                                      # can be sure the queue will support this many.
    $cmd $parallel_opts $dir/log/update.$x.log \
      ivector-extractor-est --num-threads=$nt $dir/$x.ie $dir/acc.$x $dir/$[$x+1].ie || exit 1;
    rm $dir/acc.$x.*
    $cleanup && rm $dir/acc.$x $dir/$x.ie
  fi
  x=$[$x+1]
done
$cleanup && rm -f $dir/post.*.gz
rm -f $dir/final.ie
ln -s $x.ie $dir/final.ie
