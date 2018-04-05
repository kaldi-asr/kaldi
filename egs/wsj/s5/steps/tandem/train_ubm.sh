#!/bin/bash
# Copyright 2012  Johns Hopkins University (Author: Daniel Povey).  Apache 2.0.

# This trains a UBM (i.e. a mixture of Gaussians), by clustering
# the Gaussians from a trained HMM/GMM system and then doing a few
# iterations of UBM training.
# We mostly use this for SGMM systems.

# Begin configuration section.
nj=4
cmd=run.pl
silence_weight=  # You can set it to e.g. 0.0, to weight down silence in training.
stage=-2
num_gselect1=50 # first stage of Gaussian-selection
num_gselect2=25 # second stage.
intermediate_num_gauss=2000
num_iters=3
# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;


if [ $# != 6 ]; then
  echo "Usage: steps/train_tandem_ubm.sh <num-gauss> <data1> <data2> <lang> <ali-dir> <exp>"
  echo " e.g.: steps/train_tandem_ubm.sh 400 {mfcc,bottneneck}/data/train_si84 data/lang exp/tri2b_ali_si84 exp/ubm3c"
  echo "main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config containing options"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --silence-weight <sil-weight>                    # weight for silence (e.g. 0.5 or 0.0)"
  echo "  --num-iters <#iters>                             # Number of iterations of E-M"
  exit 1;
fi

num_gauss=$1
data1=$2
data2=$3
lang=$4
alidir=$5
dir=$6

for f in $data1/feats.scp $data2/feats.scp $lang/L.fst $alidir/ali.1.gz $alidir/final.mdl; do
  [ ! -f $f ] && echo "No such file $f" && exit 1;
done

if [ $[$num_gauss*2] -gt $intermediate_num_gauss ]; then
  echo "intermediate_num_gauss was too small $intermediate_num_gauss"
  intermediate_num_gauss=$[$num_gauss*2];
  echo "setting it to $intermediate_num_gauss"
fi


# Set various variables.
silphonelist=`cat $lang/phones/silence.csl` || exit 1;
nj=`cat $alidir/num_jobs` || exit 1;

mkdir -p $dir/log
echo $nj > $dir/num_jobs

utils/lang/check_phones_compatible.sh $lang/phones.txt $alidir/phones.txt || exit 1;
cp $lang/phones.txt $dir || exit 1;

sdata1=$data1/split$nj;
sdata2=$data2/split$nj;

[[ -d $sdata1 && $data1/feats.scp -ot $sdata1 ]] || split_data.sh $data1 $nj || exit 1;
[[ -d $sdata2 && $data2/feats.scp -ot $sdata2 ]] || split_data.sh $data2 $nj || exit 1;

splice_opts=`cat $alidir/splice_opts 2>/dev/null` # frame-splicing options.
normft2=`cat $alidir/normft2 2>/dev/null`

## Set up features.

if [ -f $alidir/final.mat ]; then feat_type=lda; else feat_type=delta; fi

case $feat_type in
  delta)
    echo "$0: feature type is $feat_type"
    ;;
  lda)
    echo "$0: feature type is $feat_type"
    cp $alidir/{lda,final}.mat $dir/ || exit 1;
    ;;
  *) echo "$0: invalid feature type $feat_type" && exit 1;
esac

# set up feature stream 1;  this are usually spectral features, so we will add
# deltas or splice them
feats1="ark,s,cs:apply-cmvn --norm-vars=false --utt2spk=ark:$sdata1/JOB/utt2spk scp:$sdata1/JOB/cmvn.scp scp:$sdata1/JOB/feats.scp ark:- |"

if [ "$feat_type" == "delta" ]; then
  feats1="$feats1 add-deltas ark:- ark:- |"
elif [ "$feat_type" == "lda" ]; then
  feats1="$feats1 splice-feats $splice_opts ark:- ark:- | transform-feats $dir/lda.mat ark:- ark:- |"
fi

# set up feature stream 2;  this are usually bottleneck or posterior features,
# which may be normalized if desired
feats2="scp:$sdata2/JOB/feats.scp"

if [ "$normft2" == "true" ]; then
  feats2="ark,s,cs:apply-cmvn --norm-vars=false --utt2spk=ark:$sdata2/JOB/utt2spk scp:$sdata2/JOB/cmvn.scp $feats2 ark:- |"
fi

# assemble tandem features
feats="ark,s,cs:paste-feats '$feats1' '$feats2' ark:- |"

# add transformation, if applicable
if [ "$feat_type" == "lda" ]; then
  feats="$feats transform-feats $dir/final.mat ark:- ark:- |"
fi

# splicing/normalization options
cp $alidir/{splice_opts,tandem,normft2} $dir 2>/dev/null

if [ -f $alidir/trans.1 ]; then
  echo "$0: using transforms from $alidir"
  feats="$feats transform-feats --utt2spk=ark:$sdata1/JOB/utt2spk ark,s,cs:$alidir/trans.JOB ark:- ark:- |"
fi
##

if [ ! -z "$silence_weight" ]; then
  weights_opt="--weights='ark,s,cs:gunzip -c $alidir/ali.JOB.gz | ali-to-post ark:- ark:- | weight-silence-post $silence_weight $silphonelist $alidir/final.mdl ark:- ark:- | post-to-weights ark:- ark:- |'"
else
  weights_opt=
fi

if [ $stage -le -2 ]; then
  echo "$0: clustering model $alidir/final.mdl to get initial UBM"
  $cmd $dir/log/cluster.log \
    init-ubm --intermediate-num-gauss=$intermediate_num_gauss --ubm-num-gauss=$num_gauss \
    --verbose=2 --fullcov-ubm=true $alidir/final.mdl $alidir/final.occs \
    $dir/0.ubm   || exit 1;
fi

# Do initial phase of Gaussian selection and save it to disk -- later on we'll
# do more Gaussian selection to further prune, as the model changes.


if [ $stage -le -1 ]; then
  echo "$0: doing Gaussian selection"
  $cmd JOB=1:$nj $dir/log/gselect.JOB.log \
    gmm-gselect --n=$num_gselect1 "fgmm-global-to-gmm $dir/0.ubm - |" "$feats" \
    "ark:|gzip -c >$dir/gselect.JOB.gz" || exit 1;
fi


x=0
while [ $x -lt $num_iters ]; do
  echo "Pass $x"
  $cmd JOB=1:$nj $dir/log/acc.$x.JOB.log \
    gmm-gselect --n=$num_gselect2 "--gselect=ark,s,cs:gunzip -c $dir/gselect.JOB.gz|" \
    "fgmm-global-to-gmm $dir/$x.ubm - |" "$feats" ark:- \| \
    fgmm-global-acc-stats $weights_opt --gselect=ark,s,cs:- $dir/$x.ubm "$feats" \
    $dir/$x.JOB.acc || exit 1;
  lowcount_opt="--remove-low-count-gaussians=false"
  [ $[$x+1] -eq $num_iters ] && lowcount_opt=   # Only remove low-count Gaussians
  # on last iter-- we can't do it earlier, or the Gaussian-selection info would
  # be mismatched.
  $cmd $dir/log/update.$x.log \
    fgmm-global-est $lowcount_opt --verbose=2 $dir/$x.ubm "fgmm-global-sum-accs - $dir/$x.*.acc |" \
      $dir/$[$x+1].ubm || exit 1;
  rm $dir/$x.*.acc $dir/$x.ubm
  x=$[$x+1]
done

rm $dir/gselect.*.gz
rm $dir/final.ubm 2>/dev/null
mv $dir/$x.ubm $dir/final.ubm || exit 1;
