#!/bin/bash

# This trains a diagonal-covariance UBM (i.e. just a global
# mixture of Gaussians, or GMM).

# Train UBM from a trained HMM/GMM system [with splice+LDA+[MLLT/ET/MLLT+SAT] features]
# Alignment directory is used for the CMN and transforms.
# A UBM is just a single mixture of Gaussians (full-covariance, in our case), that's trained
# on all the data.  This will later be used in Subspace Gaussian Mixture Model (SGMM)
# training.

nj=4
cmd=scripts/run.pl
silweight=
for x in 1 2; do
  if [ $1 == "--num-jobs" ]; then
     shift
     nj=$1
     shift
  fi
  if [ $1 == "--cmd" ]; then
     shift
     cmd=$1
     shift
  fi  
  if [ $1 == "--silence-weight" ]; then
     shift
     silweight=$1 # e.g. to weight down silence in training.
     shift
  fi  
done

if [ $# != 5 ]; then
  echo "Usage: steps/train_ubm_lda_etc.sh <num-comps> <data-dir> <lang-dir> <ali-dir> <exp-dir>"
  echo " e.g.: steps/train_ubm_lda_etc.sh 400 data/train_si84 data/lang exp/tri2b_ali_si84 exp/ubm3c"
  exit 1;
fi

if [ -f path.sh ]; then . path.sh; fi

numcomps=$1
data=$2
lang=$3
alidir=$4
dir=$5
silphonelist=`cat $lang/silphones.csl`

mkdir -p $dir/log

if [ ! -d $data/split$nj -o $data/split$nj -ot $data/feats.scp ]; then
  scripts/split_data.sh $data $nj
fi

n1=`get_splits.pl $nj | awk '{print $1}'`
[ -f $alidir/$n1.trans ] && echo "Using speaker transforms from $alidir"

for n in `get_splits.pl $nj`; do
  featspart[$n]="ark,s,cs:apply-cmvn --norm-vars=false --utt2spk=ark:$data/split$nj/$n/utt2spk ark:$alidir/$n.cmvn scp:$data/split$nj/$n/feats.scp ark:- | splice-feats ark:- ark:- | transform-feats $alidir/final.mat ark:- ark:- |"
  if [ -f $alidir/$n1.trans ]; then
    featspart[$n]="${featspart[$n]} transform-feats --utt2spk=ark:$data/split$nj/$n/utt2spk ark:$alidir/$n.trans ark:- ark:- |"
  fi
  if [ ! -z "$silweight" ]; then
    weightspart[$n]="--weights='ark,s,cs:gunzip -c $alidir/$n.ali.gz | ali-to-post ark:- ark:- | weight-silence-post $silweight $silphonelist $alidir/final.mdl ark:- ark:- | post-to-weights ark:- ark:- |'"
  fi
done

ngselect=50

intermediate=2000
if [ $[$numcomps*2] -gt $intermediate ]; then
  intermediate=$[$numcomps*2];
fi

echo "Clustering model $alidir/final.mdl to get initial UBM"
# typically: --intermediate-numcomps=2000 --ubm-numcomps=400

if [ ! -s  $dir/0.dubm ]; then
 $cmd $dir/log/cluster.log \
  init-ubm --intermediate-numcomps=$intermediate --ubm-numcomps=$numcomps \
   --verbose=2 --fullcov-ubm=false $alidir/final.mdl $alidir/final.occs \
    $dir/0.dubm   || exit 1;
fi
rm $dir/.error 2>/dev/null
# First do Gaussian selection to 50 components, which will be used
# as the initial screen for all further passes.
for n in `get_splits.pl $nj`; do
  $cmd $dir/log/gselect.$n.log \
    gmm-gselect --n=$ngselect $dir/0.dubm "${featspart[$n]}" \
      "ark:|gzip -c >$dir/gselect.$n.gz"  &
done
wait
[ -f $dir/.error ] && echo "Error doing GMM selection" && exit 1;

for x in 0 1 2 3; do
  echo "Pass $x"
  for n in `get_splits.pl $nj`; do
    $cmd $dir/log/acc.$x.$n.log \
      gmm-global-acc-stats ${weightspart[$n]} "--gselect=ark,s,cs:gunzip -c $dir/gselect.$n.gz|" \
        $dir/$x.dubm "${featspart[$n]}" $dir/$x.$n.acc || touch $dir/.error &
  done
  wait
  [ -f $dir/.error ] && echo "Error accumulating stats for UBM estimation on pass $x" && exit 1;
  lowcount_opt="--remove-low-count-gaussians=false"
  [ $x -eq 3 ] && lowcount_opt=   # Only remove low-count Gaussians on last iter-- keeps gselect info valid.
  $cmd $dir/log/update.$x.log \
    gmm-global-est $lowcount_opt --verbose=2 $dir/$x.dubm "gmm-global-sum-accs - $dir/$x.*.acc |" \
      $dir/$[$x+1].dubm || exit 1;
  rm $dir/$x.*.acc $dir/$x.dubm
done

rm $dir/gselect.*.gz
rm $dir/final.dubm 2>/dev/null
mv $dir/4.dubm $dir/final.dubm || exit 1;

