#!/bin/bash
# Copyright 2012  Johns Hopkins University (Author: Daniel Povey).  Apache 2.0.


# This does Speaker Adapted Training (SAT), i.e. train on
# fMLLR-adapted features.  It can be done on top of either LDA+MLLT, or
# delta and delta-delta features.  If there are no transforms supplied
# in the alignment directory, it will estimate transforms itself before
# building the tree (and in any case, it estimates transforms a number
# of times during training).


# Begin configuration section.
stage=-5
cmd=run.pl
scale_opts="--transition-scale=1.0 --acoustic-scale=0.1 --self-loop-scale=0.1"
beam=10
retry_beam=40
boost_silence=1.0 # Factor by which to boost silence likelihoods in alignment
context_opts=  # e.g. set this to "--context-width 5 --central-position 2" for quinphone.
#realign_iters="10 20 30";
realign_iters="";
silence_weight=0.0 # Weight on silence in fMLLR estimation.
num_iters=35   # Number of iterations of training
max_iter_inc=25 # Last iter to increase #Gauss on.
power=0.2 # Exponent for number of gaussians according to occurrence counts
cluster_thresh=0  # for build-tree control final bottom-up clustering of leaves
phone_map=
train_tree=true
numtrees=1
lambda=0
# End configuration section.

echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh
. parse_options.sh || exit 1;

if [ $# != 6 ]; then
  echo "Usage: steps/train_sat.sh <#leaves> <#gauss> <data> <lang> <ali-dir> <exp-dir>"
  echo " e.g.: steps/train_sat.sh 2500 15000 data/train_si84 data/lang exp/tri2b_ali_si84 exp/tri3b"
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

for f in $data/feats.scp $lang/phones.txt $alidir/final.mdl $alidir/ali.1.gz; do
  [ ! -f $f ] && echo "train_sat.sh: no such file $f" && exit 1;
done

numgauss=$numleaves
incgauss=$[($totgauss-$numgauss)/$max_iter_inc]  # per-iter #gauss increment
oov=`cat $lang/oov.int`
nj=`cat $alidir/num_jobs` || exit 1;
silphonelist=`cat $lang/phones/silence.csl`
ciphonelist=`cat $lang/phones/context_indep.csl` || exit 1;
sdata=$data/split$nj;
splice_opts=`cat $alidir/splice_opts 2>/dev/null` # frame-splicing options.
cmvn_opts=`cat $alidir/cmvn_opts 2>/dev/null`
phone_map_opt=
[ ! -z "$phone_map" ] && phone_map_opt="--phone-map='$phone_map'"

mkdir -p $dir/log
cp $alidir/splice_opts $dir 2>/dev/null # frame-splicing options.
cp $alidir/cmvn_opts $dir 2>/dev/null # cmn/cmvn option.

echo $nj >$dir/num_jobs
[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1;

# Set up features.

if [ -f $alidir/final.mat ]; then feat_type=lda; else feat_type=delta; fi
echo "$0: feature type is $feat_type"

## Set up speaker-independent features.
case $feat_type in
  delta) sifeats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | add-deltas ark:- ark:- |";;
  lda) sifeats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | splice-feats $splice_opts ark:- ark:- | transform-feats $alidir/final.mat ark:- ark:- |"
    cp $alidir/final.mat $dir    
    cp $alidir/full.mat $dir 2>/dev/null
    ;;
  *) echo "$0: invalid feature type $feat_type" && exit 1;
esac

## Get initial fMLLR transforms (possibly from alignment dir)
if [ -f $alidir/trans.1 ]; then
  echo "$0: Using transforms from $alidir"
  feats="$sifeats transform-feats --utt2spk=ark:$sdata/JOB/utt2spk ark,s,cs:$alidir/trans.JOB ark:- ark:- |"
  cur_trans_dir=$alidir
else 
  echo this is bad!!!
fi

if [ $stage -le -4 ] && $train_tree; then
  # Get tree stats.
  echo "$0: Accumulating tree stats"
  $cmd JOB=1:$nj $dir/log/acc_tree.JOB.log \
    acc-tree-stats $context_opts --ci-phones=$ciphonelist $alidir/final.mdl "$feats" \
    "ark:gunzip -c $alidir/ali.JOB.gz|" $dir/JOB.treeacc || exit 1;
  [ "`ls $dir/*.treeacc | wc -w`" -ne "$nj" ] && echo "$0: Wrong #tree-accs" && exit 1;
  $cmd $dir/log/sum_tree_acc.log \
    sum-tree-stats $dir/treeacc $dir/*.treeacc || exit 1;
#  rm $dir/*.treeacc
fi

if [ $stage -le -3 ] && $train_tree; then
  echo "$0: Getting questions for tree clustering."
  # preparing questions, roots file...
  cluster-phones $context_opts $dir/treeacc $lang/phones/sets.int $dir/questions.int 2> $dir/log/questions.log || exit 1;
  cat $lang/phones/extra_questions.int >> $dir/questions.int
  compile-questions $context_opts $lang/topo $dir/questions.int $dir/questions.qst 2>$dir/log/compile_questions.log || exit 1;

  echo "$0: Building the tree"
  $cmd $dir/log/build_tree.log \
    build-tree-entropy $context_opts --verbose=1 --max-leaves=$numleaves \
    --num-trees=$numtrees  --lambda=$lambda \
    --thresh=0 \
    --cluster-thresh=$cluster_thresh $dir/treeacc $lang/phones/roots.int \
    $dir/questions.qst $lang/topo $dir/tree || exit 1;
fi


if [ $stage -le -2 ]; then
  echo "$0: Initializing the model"

  for i in `seq 0 $[numtrees-1]`; do
    mkdir -p $dir/tree_$i
    echo $nj > $dir/tree_$i/num_jobs
    mkdir -p $dir/tree_$i/log
    cp $dir/tree-$i $dir/tree_$i/tree   #  used for alignment in dnn .... 
    gmm-init-model  --write-occs=$dir/tree_$i/1.occs  \
      $dir/tree-$i $dir/treeacc $lang/topo $dir/tree_$i/1.mdl 2> $dir/tree_$i/log/init_model.log || exit 1;
    grep 'no stats' $dir/tree_$i/log/init_model.log && echo "This is a bad warning.";
#    rm $dir/treeacc  # not now
  done
fi

if [ $stage -le -1 ]; then
  # Convert the alignments.
  echo "$0: Converting alignments from $alidir to use current tree"
  for i in `seq 0 $[numtrees-1]`; do 
    $cmd JOB=1:$nj $dir/tree_$i/log/convert.JOB.log \
      convert-ali $phone_map_opt $alidir/final.mdl $dir/tree_$i/1.mdl $dir/tree-$i \
       "ark:gunzip -c $alidir/ali.JOB.gz|" "ark:|gzip -c >$dir/tree_$i/ali.JOB.gz" || exit 1;
  done
fi

for i in `seq 0 $[numtrees-1]`; do 
  (
    x=1
    while [ $x -lt $num_iters ]; do
       echo Pass $x

      if [ $stage -le $x ]; then
        $cmd JOB=1:$nj $dir/tree_$i/log/acc.$x.JOB.log \
      gmm-acc-stats-ali $dir/tree_$i/$x.mdl "$feats" \
      "ark,s,cs:gunzip -c $dir/tree_$i/ali.JOB.gz|" $dir/tree_$i/$x.JOB.acc || exit 1;
        [ `ls $dir/tree_$i/$x.*.acc | wc -w` -ne "$nj" ] && echo "$0: Wrong #accs" && exit 1;
        $cmd $dir/tree_$i/log/update.$x.log \
      gmm-est --power=$power --write-occs=$dir/tree_$i/$[$x+1].occs --mix-up=$numgauss $dir/tree_$i/$x.mdl \
      "gmm-sum-accs - $dir/tree_$i/$x.*.acc |" $dir/tree_$i/$[$x+1].mdl || exit 1;
# now deleting the accs
      rm $dir/tree_$i/$x.mdl $dir/tree_$i/$x.*.acc
      rm $dir/tree_$i/$x.occs 
      fi
      [ $x -le $max_iter_inc ] && numgauss=$[$numgauss+$incgauss];
      x=$[$x+1];
    done
    touch $dir/done.$i
  )
done


while [ `ls $dir/done* 2>/dev/null | wc -l` -ne $numtrees ]; do sleep 30; done

if [ `ls $dir/tree_*/${num_iters}.mdl | wc -l` -ne $numtrees ]; then
    echo something is wrong
    exit 1
fi

echo training for all tree done
rm $dir/done.*

x=$num_iters

for i in `seq 0 $[numtrees-1]`; do
  if [ $stage -le $x ]; then
    # Accumulate stats for "alignment model"-- this model is
    # computed with the speaker-independent features, but matches Gaussian-for-Gaussian
    # with the final speaker-adapted model.
    $cmd JOB=1:$nj $dir/tree_$i/log/acc_alimdl.JOB.log \
      ali-to-post "ark:gunzip -c $dir/tree_$i/ali.JOB.gz|" ark:-  \| \
      gmm-acc-stats-twofeats $dir/tree_$i/$x.mdl "$feats" "$sifeats" \
      ark,s,cs:- $dir/tree_$i/$x.JOB.acc || exit 1;
    [ `ls $dir/tree_$i/$x.*.acc | wc -w` -ne "$nj" ] && echo "$0: Wrong #accs" && exit 1;
    # Update model.
    $cmd $dir/tree_$i/log/est_alimdl.log \
      gmm-est --power=$power --remove-low-count-gaussians=false $dir/tree_$i/$x.mdl \
      "gmm-sum-accs - $dir/tree_$i/$x.*.acc|" $dir/tree_$i/$x.alimdl  || exit 1;
#    rm $dir/tree_$i/$x.*.acc
  fi

  rm $dir/tree_$i/final.{mdl,alimdl,occs} 2>/dev/null
  ln -s $x.mdl $dir/tree_$i/final.mdl
  ln -s $x.occs $dir/tree_$i/final.occs
  ln -s $x.alimdl $dir/tree_$i/final.alimdl
done

for i in `seq 0 $[numtrees-1]`; do 
  utils/summarize_warnings.pl $dir/tree_$i/log
  (
    echo "$0: Likelihood evolution:"
    for x in `seq $[$num_iters-1]`; do
      tail -n 30 $dir/tree_$i/log/acc.$x.*.log | awk '/Overall avg like/{l += $(NF-3)*$(NF-1); t += $(NF-1); }
	  /Overall average logdet/{d += $(NF-3)*$(NF-1); t2 += $(NF-1);} 
	  END{ d /= t2; l /= t; printf("%s ", d+l); } '
    done
    echo
  ) | tee $dir/tree_$i/log/summary.log
done
echo Done
