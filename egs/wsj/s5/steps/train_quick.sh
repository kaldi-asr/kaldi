#!/bin/bash
# Copyright 2012  Johns Hopkins University (Author: Daniel Povey).  Apache 2.0.


# Train a model on top of existing features (no feature-space learning of any
# kind is done).  This script initializes the model (i.e., the GMMs) from the
# previous system's model.  That is: for each state in the current model (after
# tree building), it chooses the closes state in the old model, judging the
# similarities based on overlap of counts in the tree stats.

# Begin configuration..
cmd=run.pl
scale_opts="--transition-scale=1.0 --acoustic-scale=0.1 --self-loop-scale=0.1"
realign_iters="10 15"; # Only realign twice.
num_iters=20    # Number of iterations of training
maxiterinc=15 # Last iter to increase #Gauss on.
batch_size=750 # batch size to use while compiling graphs... memory/speed tradeoff.
beam=10 # alignment beam.
retry_beam=40
stage=-5
cluster_thresh=-1  # for build-tree control final bottom-up clustering of leaves
# End configuration section.

echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh
. parse_options.sh || exit 1;

if [ $# != 6 ]; then
  echo "Usage: steps/train_quick.sh <num-leaves> <num-gauss> <data> <lang> <ali-dir> <exp-dir>"
  echo " e.g.: steps/train_quick.sh 2500 15000 data/train_si284 data/lang exp/tri3c_ali_si284 exp/tri4b"
  echo "Main options (for others, see top of script file)"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --config <config-file>                           # config containing options"
  echo "  --stage <stage>                                  # stage to do partial re-run from."
  exit 1;
fi

numleaves=$1
totgauss=$2
data=$3
lang=$4
alidir=$5
dir=$6

for f in $data/feats.scp $lang/L.fst $alidir/ali.1.gz $alidir/final.mdl; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

# Set various variables.
oov=`cat $lang/oov.int`
silphonelist=`cat $lang/phones/silence.csl`
ciphonelist=`cat $lang/phones/context_indep.csl`
numgauss=$[totgauss/2] # Start with half the total number of Gaussians.  We won't have
  # to mix up much probably, as we're initializing with the old (already mixed-up) pdf's.  
[ $numgauss -lt $numleaves ] && numgauss=$numleaves
incgauss=$[($totgauss-$numgauss)/$maxiterinc] # per-iter increment for #Gauss
nj=`cat $alidir/num_jobs` || exit 1;
sdata=$data/split$nj
splice_opts=`cat $alidir/splice_opts 2>/dev/null` # frame-splicing options.
cmvn_opts=`cat $alidir/cmvn_opts 2>/dev/null`
delta_opts=`cat $alidir/delta_opts 2>/dev/null`

mkdir -p $dir/log
echo $nj >$dir/num_jobs
cp $alidir/splice_opts $dir 2>/dev/null
cp $alidir/cmvn_opts $dir 2>/dev/null # cmn/cmvn option.
cp $alidir/delta_opts $dir 2>/dev/null
[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1;

utils/lang/check_phones_compatible.sh $lang/phones.txt $alidir/phones.txt || exit 1;
cp $lang/phones.txt $dir || exit 1;

## Set up features.
if [ -f $alidir/final.mat ]; then feat_type=lda; else feat_type=delta; fi
echo "$0: feature type is $feat_type"

case $feat_type in
  delta) sifeats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | add-deltas $delta_opts ark:- ark:- |";;
  lda) sifeats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | splice-feats $splice_opts ark:- ark:- | transform-feats $alidir/final.mat ark:- ark:- |"
    cp $alidir/final.mat $dir    
    cp $alidir/full.mat $dir 2>/dev/null
    ;;
  *) echo "Invalid feature type $feat_type" && exit 1;
esac
if [ -f $alidir/trans.1 ]; then
  echo "$0: using transforms from $alidir"
  ln.pl $alidir/trans.* $dir # Link them to dest dir.
  feats="$sifeats transform-feats --utt2spk=ark:$sdata/JOB/utt2spk ark,s,cs:$dir/trans.JOB ark:- ark:- |"
else
  feats="$sifeats"
fi
##


if [ $stage -le -5 ]; then
  echo "$0: accumulating tree stats"
  $cmd JOB=1:$nj $dir/log/acc_tree.JOB.log \
    acc-tree-stats  --ci-phones=$ciphonelist $alidir/final.mdl "$feats" \
    "ark:gunzip -c $alidir/ali.JOB.gz|" $dir/JOB.treeacc || exit 1;
  [ "`ls $dir/*.treeacc | wc -w`" -ne "$nj" ] && echo "$0: Wrong #tree-stats" && exit 1;
  sum-tree-stats $dir/treeacc $dir/*.treeacc 2>$dir/log/sum_tree_acc.log || exit 1;
  rm $dir/*.treeacc
fi

if [ $stage -le -4 ]; then
  echo "$0: Getting questions for tree clustering."
  # preparing questions, roots file...
  cluster-phones $dir/treeacc $lang/phones/sets.int $dir/questions.int 2> $dir/log/questions.log || exit 1;
  cat $lang/phones/extra_questions.int >> $dir/questions.int
  compile-questions $lang/topo $dir/questions.int $dir/questions.qst 2>$dir/log/compile_questions.log || exit 1;

  echo "$0: Building the tree"
  $cmd $dir/log/build_tree.log \
    build-tree --verbose=1 --max-leaves=$numleaves \
    --cluster-thresh=$cluster_thresh $dir/treeacc $lang/phones/roots.int \
    $dir/questions.qst $lang/topo $dir/tree || exit 1;
fi

if [ $stage -le -3 ]; then
  echo "$0: Initializing the model"

  # The gmm-init-model command (with more than the normal # of command-line args)
  # will initialize the p.d.f.'s to the p.d.f.'s in the alignment model.

  gmm-init-model  --write-occs=$dir/1.occs  \
    $dir/tree $dir/treeacc $lang/topo $dir/tmp.mdl $alidir/tree $alidir/final.mdl  \
    2>$dir/log/init_model.log || exit 1;

  grep 'no stats' $dir/log/init_model.log && echo "$0: This is a bad warning.";
  rm $dir/treeacc
fi

if [ $stage -le -2 ]; then
  echo "$0: mixing up old model."
  # We do both mixing-down and mixing-up to get the target #Gauss in each state,
  # since the initial model may have either more or fewer Gaussians than we want.
  gmm-mixup --mix-down=$numgauss --mix-up=$numgauss $dir/tmp.mdl $dir/1.occs $dir/1.mdl \
    2> $dir/log/mixup.log || exit 1;
  rm $dir/tmp.mdl 
fi

# Convert alignments to the new tree.
if [ $stage -le -1 ]; then
  echo "$0: converting old alignments"
  $cmd JOB=1:$nj $dir/log/convert.JOB.log \
    convert-ali $alidir/final.mdl $dir/1.mdl $dir/tree \
    "ark:gunzip -c $alidir/ali.JOB.gz|" "ark:|gzip -c >$dir/ali.JOB.gz" || exit 1;
fi

if [ $stage -le 0 ]; then
  echo "$0: compiling training graphs"
  $cmd JOB=1:$nj $dir/log/compile_graphs.JOB.log \
    compile-train-graphs --read-disambig-syms=$lang/phones/disambig.int --batch-size=$batch_size $dir/tree $dir/1.mdl $lang/L.fst  \
    "ark:sym2int.pl --map-oov $oov -f 2- $lang/words.txt < $sdata/JOB/text |" \
    "ark:|gzip -c >$dir/fsts.JOB.gz" || exit 1;
fi

x=1
while [ $x -lt $num_iters ]; do
  echo "$0: pass $x"
  if echo $realign_iters | grep -w $x >/dev/null && [ $stage -le $x ]; then
    echo "$0: aligning data"
    $cmd JOB=1:$nj $dir/log/align.$x.JOB.log \
      gmm-align-compiled $scale_opts --beam=$beam --retry-beam=$retry_beam $dir/$x.mdl \
      "ark:gunzip -c $dir/fsts.JOB.gz|" "$feats" "ark:|gzip -c >$dir/ali.JOB.gz" \
      || exit 1;
  fi
  if [ $stage -le $x ]; then
    $cmd JOB=1:$nj $dir/log/acc.$x.JOB.log \
      gmm-acc-stats-ali  $dir/$x.mdl "$feats" \
      "ark,s,cs:gunzip -c $dir/ali.JOB.gz|"  $dir/$x.JOB.acc || exit 1;
    [ "`ls $dir/$x.*.acc | wc -w`" -ne "$nj" ] && echo "$0: wrong #accs" && exit 1;
    $cmd $dir/log/update.$x.log \
      gmm-est --write-occs=$dir/$[$x+1].occs --mix-up=$numgauss $dir/$x.mdl \
      "gmm-sum-accs - $dir/$x.*.acc |" $dir/$[$x+1].mdl || exit 1;
    rm $dir/$x.mdl $dir/$x.*.acc $dir/$x.occs
  fi
  [[ $x -le $maxiterinc ]] && numgauss=$[$numgauss+$incgauss];
  x=$[$x+1];
done

if [ -f $alidir/trans.1 ]; then
  echo "$0: estimating alignment model"
  $cmd JOB=1:$nj $dir/log/acc_alimdl.JOB.log \
    ali-to-post "ark:gunzip -c $dir/ali.JOB.gz|" ark:-  \| \
    gmm-acc-stats-twofeats $dir/$x.mdl "$feats" "$sifeats" \
    ark,s,cs:- $dir/$x.JOB.acc || exit 1;
  [ "`ls $dir/$x.*.acc | wc -w`" -ne "$nj" ] && echo "$0: wrong #accs" && exit 1;

  $cmd $dir/log/est_alimdl.log \
    gmm-est --write-occs=$dir/final.occs --remove-low-count-gaussians=false $dir/$x.mdl \
    "gmm-sum-accs - $dir/$x.*.acc|" $dir/$x.alimdl || exit 1;
  rm $dir/$x.*.acc
  rm $dir/final.alimdl 2>/dev/null 
  ln -s $x.alimdl $dir/final.alimdl
fi

rm $dir/final.mdl 2>/dev/null
ln -s $x.mdl $dir/final.mdl

echo Done
