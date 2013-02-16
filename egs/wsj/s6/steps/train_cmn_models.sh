#!/bin/bash
# Copyright 2012  Johns Hopkins University (Author: Daniel Povey).  Apache 2.0.

# This trains two diagonal, global Gaussian Mixture Models (we call them UBMs
# here but the term is kind of mis-applied), by clustering
# the Gaussians from a trained HMM/GMM system and then doing a few
# iterations of UBM training; and then training each one on either 
# speech or silence.
# It then computes some stats that will be used for a special "balanced" CMVN
# computation, by the script compute_cmvn_stats_balanced.sh

# This recipe is intended for use in a situation where each speech/sil
# UBM is quite small.

# Begin configuration section.
nj=4
cmd=run.pl
silence_weight=1.0 # applies to LDA and MLLT estimation.  We don't 
                   # down-weight silence, since we're interested in it.
stage=-7
splice_opts="--left-context=5 --right-context=6" # Use a lot of frames,
  # as we'll just be making per-frame judgements; this should be more accurate.
num_gauss=80
num_gselect=20 # Gaussian-selection
intermediate_num_gauss=1000
real_silence_phones=  # this will default to the contents of $lang/phones/optional_silence.csl
  # but can be set by the user.
num_iters=3
cleanup=true
dim=40 # LDA dimension
randprune=4.0 # This is approximately the ratio by which we will speed up the
              # LDA and MLLT calculations via randomized pruning.
# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;


if [ $# != 4 ]; then
  echo "Usage: steps/train_speechsil_dubms.sh <data> <lang> <ali-dir> <exp>"
  echo " e.g.: steps/train_speechsil_dubms.sh data/train_si84 data/lang exp/tri2b_ali_si84 exp/ubm3c"
  echo "main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config containing options"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --num-iters <#iters>                             # Number of iterations of E-M before final speech/sil split, default 3"
  echo "  --num-gauss <#gauss>                             # Number of Gaussians, default 50."
  echo "  --num-gselect <#gselect>                         # Number of Gaussians pre-selected per frame, default 20."
  echo "  --cleanup <true|false>                           # Clean up intermediate models etc.; default true"
  echo "  --use-fmllr <true|false>                         # Use any fMLLR transforms in alignment dir; default true"
  echo "  --real-silence-phones <comma-separated-integers> # List of integer id's of silence phones that you expect"
  echo "                                                   # will be actually silent.  Defaults to contents of"
  echo "                                                   # lang/phones/optional_silence.csl"
  exit 1;
fi

data=$1
lang=$2
alidir=$3
dir=$4

for f in $data/feats.scp $lang/L.fst $alidir/ali.1.gz $alidir/final.mdl; do
  [ ! -f $f ] && echo "No such file $f" && exit 1;
done

if [ $[$num_gauss*2] -gt $intermediate_num_gauss ]; then
  echo "intermediate_num_gauss was too small $intermediate_num_gauss"
  intermediate_num_gauss=$[$num_gauss*2];
  echo "setting it to $intermediate_num_gauss"
fi


# Set various variables.
silphonelist=`cat $lang/phones/silence.csl` || exit 1;
if [ -z "$real_silence_phones" ]; then
  real_silence_phones=`cat $lang/phones/optional_silence.csl` || exit 1;
fi
nj=`cat $alidir/num_jobs` || exit 1;

mkdir -p $dir/log
echo $nj > $dir/num_jobs
sdata=$data/split$nj;
[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1;


splicedfeats="ark,s,cs:splice-feats $splice_opts scp:$sdata/JOB/feats.scp ark:- |"
echo "$splice_opts" >$dir/splice_opts # keep track of frame-splicing options

feats="$splicedfeats transform-feats $dir/0.mat ark:- ark:- |" # We'll change this later
# to use final.mat

if [ $stage -le -7 ]; then
  echo "Accumulating LDA statistics."
  $cmd JOB=1:$nj $dir/log/lda_acc.JOB.log \
    ali-to-post "ark:gunzip -c $alidir/ali.JOB.gz|" ark:- \| \
      weight-silence-post $silence_weight $silphonelist $alidir/final.mdl ark:- ark:- \| \
      acc-lda --rand-prune=$randprune $alidir/final.mdl "$splicedfeats" ark,s,cs:- \
       $dir/lda.JOB.acc || exit 1;
  est-lda --dim=$dim $dir/0.mat $dir/lda.*.acc \
      2>$dir/log/lda_est.log || exit 1;
fi
$cleanup && rm $dir/lda.*.acc

if [ $stage -le -6 ]; then
  cp $alidir/tree $dir/ || exit 1;
  $cmd JOB=1 $dir/log/init_model.log \
    gmm-init-model-flat $dir/tree $lang/topo $dir/0.mdl \
    "$feats subset-feats ark:- ark:-|" || exit 1;
fi

if [ $stage -le -5 ]; then
  $cmd JOB=1:$nj $dir/log/acc_mdl.JOB.log \
    gmm-acc-stats-ali $dir/0.mdl "$feats" \
    "ark,s,cs:gunzip -c $alidir/ali.JOB.gz|" $dir/mdl.JOB.acc || exit 1;
  $cmd $dir/log/update_mdl.log \
    gmm-est --write-occs=$dir/1.occs $dir/0.mdl "gmm-sum-accs - $dir/mdl.*.acc |" $dir/1.mdl || exit 1;
fi
$cleanup && rm $dir/0.mdl $dir/mdl.*.acc 


echo "Estimating MLLT"
if [ $stage -le -4 ]; then
  $cmd JOB=1:$nj $dir/log/macc.JOB.log \
    ali-to-post "ark:gunzip -c $alidir/ali.JOB.gz|" ark:- \| \
    weight-silence-post $silence_weight $silphonelist $dir/1.mdl ark:- ark:- \| \
    gmm-acc-mllt --rand-prune=$randprune  $dir/1.mdl "$feats" ark:- $dir/JOB.macc || exit 1;
  est-mllt $dir/mllt.mat $dir/*.macc 2> $dir/log/mupdate.log || exit 1;
  gmm-transform-means  $dir/mllt.mat $dir/1.mdl $dir/2.mdl 2> $dir/log/transform_means.$x.log || exit 1;
  compose-transforms --print-args=false $dir/mllt.mat $dir/0.mat $dir/final.mat || exit 1;
fi
$cleanup && rm $dir/*.macc $dir/0.mat $dir/mllt.mat

feats="$splicedfeats transform-feats $dir/final.mat ark:- ark:- |"

if [ $stage -le -3 ]; then
  echo "$0: clustering model $dir/2.mdl to get initial UBM"
  $cmd $dir/log/cluster.log \
    init-ubm --intermediate-num-gauss=$intermediate_num_gauss --ubm-num-gauss=$num_gauss \
    --verbose=2 --fullcov-ubm=false $dir/2.mdl $dir/1.occs \
    $dir/0.ubm   || exit 1;
fi

# Do Gaussian selection and save it to disk.

echo "$0: doing Gaussian selection"
if [ $stage -le -2 ]; then
  $cmd JOB=1:$nj $dir/log/gselect.JOB.log \
    gmm-gselect --n=$num_gselect $dir/0.ubm "$feats" \
    "ark:|gzip -c >$dir/gselect.JOB.gz" || exit 1;
fi

x=0
while [ $x -lt $num_iters ]; do
  echo "Pass $x"
  if [ $stage -le $x ]; then
    $cmd JOB=1:$nj $dir/log/acc.$x.JOB.log \
      gmm-global-acc-stats "--gselect=ark,s,cs:gunzip -c $dir/gselect.JOB.gz|" $dir/$x.ubm "$feats" \
      $dir/$x.JOB.acc || exit 1;

    $cmd $dir/log/update.$x.log \
      gmm-global-est --verbose=2 $dir/$x.ubm "gmm-global-sum-accs - $dir/$x.*.acc |" \
      $dir/$[$x+1].ubm || exit 1; 
  fi
  $cleanup && rm $dir/$x.*.acc $dir/$x.ubm
  x=$[$x+1]
done

# Now one final iteration with speech and silence separately.

nonsilweights="ark,s,cs:gunzip -c $alidir/ali.JOB.gz | ali-to-post ark:- ark:- | weight-silence-post 0.0 $silphonelist $alidir/final.mdl ark:- ark:- | post-to-weights ark:- ark:- |"

echo "Final non-silence pass"

if [ $stage -le $x ]; then
  $cmd JOB=1:$nj $dir/log/acc.nonsilence.JOB.log \
    gmm-global-acc-stats "--weights=$nonsilweights" "--gselect=ark,s,cs:gunzip -c $dir/gselect.JOB.gz|" $dir/$x.ubm "$feats" \
    $dir/nonsilence.JOB.acc || exit 1;

  $cmd $dir/log/update.nonsilence.log \
    gmm-global-est --remove-low-count-gaussians=false --verbose=2 $dir/$x.ubm \
    "gmm-global-sum-accs - $dir/nonsilence.*.acc |" \
    $dir/nonsilence.ubm || exit 1;
fi
$cleanup && rm $dir/nonsilence.*.acc 

echo "Final silence pass"

# weight to one only "$real_silence_phones" which defaults to the silence in 
# $lang/phones/optional_silence.txt.  This is generally SIL or sil-- the "normal"
# silence. 
silweights="ark,s,cs:gunzip -c $alidir/ali.JOB.gz | ali-to-post ark:- ark:- | weight-silence-post 0.0 $real_silence_phones $alidir/final.mdl ark:- ark:- | post-to-weights ark:- ark:- | reverse-weights ark:- ark:- |"

if [ $stage -le $num_iters ]; then
  $cmd JOB=1:$nj $dir/log/acc.silence.JOB.log \
    gmm-global-acc-stats "--weights=$silweights" "--gselect=ark,s,cs:gunzip -c $dir/gselect.JOB.gz|" $dir/$x.ubm "$feats" \
    $dir/silence.JOB.acc || exit 1;

  $cmd $dir/log/update.silence.log \
    gmm-global-est --remove-low-count-gaussians=false --verbose=2 $dir/$x.ubm \
    "gmm-global-sum-accs - $dir/silence.*.acc |" \
    $dir/silence.ubm || exit 1;
fi
$cleanup && rm $dir/silence.*.acc $dir/$x.ubm

if [ $stage -le $[$num_iters+1] ]; then
  echo "$0: computing non-silence weights of training data."

  $cmd JOB=1:$nj $dir/log/nonsil_probs.JOB.log \
    get-silence-probs --write-nonsil-probs=true \
    "$feats gmm-global-get-frame-likes $dir/silence.ubm ark:- ark:- |" \
    "$feats gmm-global-get-frame-likes $dir/nonsilence.ubm ark:- ark:- |" \
    "ark,t:|gzip -c >$dir/weights.JOB.gz" || exit 1;
fi  


if [ $stage -le $[$num_iters+2] ]; then
  echo "$0: computing speech and silence CMVN averages of baseline"

  # This is the CMVN stats of the features that are already normalized with the
  # CMVN stats in the baseline dir.  This is so that once we work out the
  # speech/silence probabilities, we can compute a CMVN shift (and possibly
  # scale) that will give us the same stats as would have been produced by
  # the specified proportions of speech and silence.. sorry, this is a bit
  # unclear; the code should have some clearer comments in it.
  
  weights="ark:gunzip -c $dir/weights.*.gz|"

  # Note: nonsilence.cmvn is not a table/archive, it's a file, so the program will
  # compute global stats.
  ! compute-cmvn-stats --binary=false --weights="$weights" scp:$data/feats.scp $dir/nonsilence.cmvn \
    2> $dir/log/cmvn_nonsilence.log && echo "Error computing non-silence CMVN stats" && exit 1;

  weights="$weights reverse-weights ark:- ark:- |"

  ! compute-cmvn-stats --binary=false --weights="$weights" scp:$data/feats.scp $dir/silence.cmvn \
    2> $dir/log/cmvn_silence.log && echo "Error computing silence CMVN stats" && exit 1;
fi

if [ $stage -le $[$num_iters+3] ]; then
  echo "Computing accuracy of classifier";
  
  $cmd JOB=1:$nj $dir/log/dot_sil_hyp.JOB.log \
    dot-weights "ark:gunzip -c $dir/weights.JOB.gz | reverse-weights ark:- ark:- |" "$silweights" ark,t:- \| \
    awk '{x += $2;} END{printf("Dot product is %f\n", x);}' || exit 1;

  $cmd JOB=1:$nj $dir/log/dot_sil_ref.JOB.log \
    dot-weights "$silweights" "$silweights" ark,t:- \| \
    awk '{x += $2;} END{printf("Dot product is %f\n", x);}' || exit 1;

  $cmd JOB=1:$nj $dir/log/dot_nonsil_hyp.JOB.log \
    dot-weights "ark:gunzip -c $dir/weights.JOB.gz |" "$nonsilweights" ark,t:- \| \
    awk '{x += $2;} END{printf("Dot product is %f\n", x);}' || exit 1;

  $cmd JOB=1:$nj $dir/log/dot_nonsil_ref.JOB.log \
    dot-weights "$nonsilweights" "$nonsilweights" ark,t:- \| \
    awk '{x += $2;} END{printf("Dot product is %f\n", x);}' || exit 1;
fi

sil_hyp=`grep "Dot product is" $dir/log/dot_sil_hyp.*.log  | awk '{x += $NF} END{print x;}'`
sil_ref=`grep "Dot product is" $dir/log/dot_sil_ref.*.log  | awk '{x += $NF} END{print x;}'`
nonsil_hyp=`grep "Dot product is" $dir/log/dot_nonsil_hyp.*.log  | awk '{x += $NF} END{print x;}'`
nonsil_ref=`grep "Dot product is" $dir/log/dot_nonsil_ref.*.log  | awk '{x += $NF} END{print x;}'`
sil_acc=`perl -e "printf('%.2f', (100.0 * $sil_hyp / $sil_ref));"`
nonsil_acc=`perl -e "printf('%.2f', (100.0 * $nonsil_hyp / $nonsil_ref));"`

echo "Classification accuracy on training data of silence is $sil_acc%, nonsilence is $nonsil_acc%"

$cleanup && rm $dir/gselect.*.gz $dir/weights.*.gz

exit 0;
