#!/bin/bash
# Copyright 2012  Johns Hopkins University (Author: Daniel Povey).  Apache 2.0.


# This does Speaker Adapted Training (SAT).  We train on fMLLR-adapted features,
# but in this "raw" script, these transforms are at the level of the raw
# cepstra.  The model must be built on top of LDA+MLLT features, and the
# transforms are estimated using the model, in a rather clever way.  If there
# are no raw transforms supplied in the alignment directory, it will estimate
# transforms itself before building the tree (and in any case, it estimates
# transforms a number of times during training).
# You need to decode the models it builds with decode_raw_fmllr.sh

# Begin configuration section.
stage=-6
cmd=run.pl
scale_opts="--transition-scale=1.0 --acoustic-scale=0.1 --self-loop-scale=0.1"
beam=10
retry_beam=40
boost_silence=1.0 # Factor by which to boost silence likelihoods in alignment
context_opts=  # e.g. set this to "--context-width 5 --central-position 2" for quinphone.
realign_iters="10 20 30";
fmllr_iters="2 4 6 12";
mllt_iters="3 5 7 10"
dim=40
randprune=4.0 # This is approximately the ratio by which we will speed up the
              # LDA and MLLT calculations via randomized pruning.
silence_weight=0.0 # Weight on silence in fMLLR estimation.
num_iters=35   # Number of iterations of training
max_iter_inc=25 # Last iter to increase #Gauss on.
power=0.2 # Exponent for number of gaussians according to occurrence counts
cluster_thresh=-1  # for build-tree control final bottom-up clustering of leaves
train_tree=true
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
raw_dim=$(feat-to-dim scp:$data/feats.scp -) || exit 1;
! [ "$raw_dim" -gt 0 ] && echo "raw feature dim not set" && exit 1;

mkdir -p $dir/log
cp $alidir/splice_opts $dir 2>/dev/null # frame-splicing options.
cp $alidir/cmvn_opts $dir 2>/dev/null # cmn/cmvn option.

echo $nj >$dir/num_jobs
[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1;

# Set up features.

if [[ ! -f $alidir/final.mat || ! -f $alidir/full.mat ]]; then
  echo "$0: expected to find  $alidir/final.mat and $alidir/full.mat"
  exit 1
fi

sisplicedfeats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | splice-feats $splice_opts ark:- ark:- |"
sifeats="$sisplicedfeats transform-feats $alidir/final.mat ark:- ark:- |"


## Get initial fMLLR transforms (possibly from alignment dir)
if [ -f $alidir/raw_trans.1 ]; then
  echo "$0: Using transforms from $alidir"
  cur_trans_dir=$alidir
else 
  if [ $stage -le -6 ]; then
    echo "$0: obtaining initial fMLLR transforms since not present in $alidir"
    # The next line is necessary because of $silphonelist otherwise being incorrect; would require
    # old $lang dir which would require another option.  Not needed anyway.
    full_lda_mat="get-full-lda-mat --print-args=false $alidir/final.mat $alidir/full.mat -|"
    $cmd JOB=1:$nj $dir/log/fmllr.0.JOB.log \
      ali-to-post "ark:gunzip -c $alidir/ali.JOB.gz|" ark:- \| \
      weight-silence-post $silence_weight $silphonelist $alidir/final.mdl ark:- ark:- \| \
      gmm-est-fmllr-raw --raw-feat-dim=$raw_dim --spk2utt=ark:$sdata/JOB/spk2utt $alidir/final.mdl \
        "$full_lda_mat" "$sisplicedfeats" ark:- ark:$dir/raw_trans.JOB || exit 1;
  fi
  cur_trans_dir=$dir
fi

splicedfeats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | transform-feats --utt2spk=ark:$sdata/JOB/utt2spk ark,s,cs:$cur_trans_dir/raw_trans.JOB ark:- ark:- | splice-feats $splice_opts ark:- ark:- |"


if [ $stage -le -5 ]; then
  echo "Accumulating LDA statistics."
  $cmd JOB=1:$nj $dir/log/lda_acc.JOB.log \
    ali-to-post "ark:gunzip -c $alidir/ali.JOB.gz|" ark:- \| \
      weight-silence-post 0.0 $silphonelist $alidir/final.mdl ark:- ark:- \| \
      acc-lda --rand-prune=$randprune $alidir/final.mdl "$splicedfeats" ark,s,cs:- \
       $dir/lda.JOB.acc || exit 1;
  est-lda --write-full-matrix=$dir/full.mat --dim=$dim $dir/0.mat $dir/lda.*.acc \
      2>$dir/log/lda_est.log || exit 1;  
  rm $dir/lda.*.acc
fi

cur_lda_iter=0
feats="$splicedfeats transform-feats $dir/$cur_lda_iter.mat ark:- ark:- |"

# To build the tree, we use the previous directory's LDA transform, which
# is better as it has MLLT also.  It leads to higher auxiliary function
# improvements in tree building, which is generally a good thing.
tree_feats="$splicedfeats transform-feats $alidir/final.mat ark:- ark:- |"


if [ $stage -le -4 ] && $train_tree; then
  # Get tree stats.
  echo "$0: Accumulating tree stats"
  $cmd JOB=1:$nj $dir/log/acc_tree.JOB.log \
    acc-tree-stats $context_opts --ci-phones=$ciphonelist $alidir/final.mdl "$tree_feats" \
    "ark:gunzip -c $alidir/ali.JOB.gz|" $dir/JOB.treeacc || exit 1;
  [ "`ls $dir/*.treeacc | wc -w`" -ne "$nj" ] && echo "$0: Wrong #tree-accs" && exit 1;
  $cmd $dir/log/sum_tree_acc.log \
    sum-tree-stats $dir/treeacc $dir/*.treeacc || exit 1;
  rm $dir/*.treeacc
fi

if [ $stage -le -3 ] && $train_tree; then
  echo "$0: Getting questions for tree clustering."
  # preparing questions, roots file...
  cluster-phones $context_opts $dir/treeacc $lang/phones/sets.int $dir/questions.int 2> $dir/log/questions.log || exit 1;
  cat $lang/phones/extra_questions.int >> $dir/questions.int
  compile-questions $context_opts $lang/topo $dir/questions.int $dir/questions.qst 2>$dir/log/compile_questions.log || exit 1;

  echo "$0: Building the tree"
  $cmd $dir/log/build_tree.log \
    build-tree $context_opts --verbose=1 --max-leaves=$numleaves \
    --cluster-thresh=$cluster_thresh $dir/treeacc $lang/phones/roots.int \
    $dir/questions.qst $lang/topo $dir/tree || exit 1;
fi

if [ $stage -le -2 ]; then
  echo "$0: Initializing the model"
  # Since we trained the tree on different feats, we don't use gmm-init-model, which
  # would initialize the tree with invalid features.  This doesn't really matter anyway,
  # the first iteration of training will set suitable initial parameters.
  cp $alidir/tree $dir/ || exit 1;
  $cmd JOB=1 $dir/log/init_model.log \
    gmm-init-model-flat $dir/tree $lang/topo $dir/1.mdl \
    "$tree_feats subset-feats ark:- ark:-|" || exit 1;
fi

if [ $stage -le -1 ]; then
  # Convert the alignments.
  echo "$0: Converting alignments from $alidir to use current tree"
  $cmd JOB=1:$nj $dir/log/convert.JOB.log \
    convert-ali $alidir/final.mdl $dir/1.mdl $dir/tree \
     "ark:gunzip -c $alidir/ali.JOB.gz|" "ark:|gzip -c >$dir/ali.JOB.gz" || exit 1;
fi

if [ $stage -le 0 ] && [ "$realign_iters" != "" ]; then
  echo "$0: Compiling graphs of transcripts"
  $cmd JOB=1:$nj $dir/log/compile_graphs.JOB.log \
    compile-train-graphs $dir/tree $dir/1.mdl  $lang/L.fst  \
     "ark:utils/sym2int.pl --map-oov $oov -f 2- $lang/words.txt < $sdata/JOB/text |" \
      "ark:|gzip -c >$dir/fsts.JOB.gz" || exit 1;
fi

x=1
while [ $x -lt $num_iters ]; do
   echo Pass $x
  if echo $realign_iters | grep -w $x >/dev/null && [ $stage -le $x ]; then
    echo Aligning data
    mdl="gmm-boost-silence --boost=$boost_silence `cat $lang/phones/optional_silence.csl` $dir/$x.mdl - |"
    $cmd JOB=1:$nj $dir/log/align.$x.JOB.log \
      gmm-align-compiled $scale_opts --beam=$beam --retry-beam=$retry_beam "$mdl" \
      "ark:gunzip -c $dir/fsts.JOB.gz|" "$feats" \
      "ark:|gzip -c >$dir/ali.JOB.gz" || exit 1;
  fi

  if echo $fmllr_iters | grep -w $x >/dev/null; then
    if [ $stage -le $x ]; then
      echo Estimating fMLLR transforms
      # We estimate a transform that's additional to the previous transform;
      # we'll compose them.

      full_lda_mat="get-full-lda-mat --print-args=false $dir/$cur_lda_iter.mat $dir/full.mat - |"
      $cmd JOB=1:$nj $dir/log/fmllr.$x.JOB.log \
        ali-to-post "ark:gunzip -c $dir/ali.JOB.gz|" ark:-  \| \
        weight-silence-post $silence_weight $silphonelist $dir/$x.mdl ark:- ark:- \| \
        gmm-est-fmllr-raw --raw-feat-dim=$raw_dim --spk2utt=ark:$sdata/JOB/spk2utt $dir/$x.mdl "$full_lda_mat" \
          "$splicedfeats" ark:- ark:$dir/tmp_trans.JOB || exit 1;
      for n in `seq $nj`; do
        ! ( compose-transforms --b-is-affine=true \
          ark:$dir/tmp_trans.$n ark:$cur_trans_dir/raw_trans.$n ark:$dir/composed_trans.$n \
          && mv $dir/composed_trans.$n $dir/raw_trans.$n && \
          rm $dir/tmp_trans.$n ) 2>$dir/log/compose_transforms.$x.log \
          && echo "$0: Error composing transforms" && exit 1;
      done
    fi
    cur_trans_dir=$dir
    splicedfeats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | transform-feats --utt2spk=ark:$sdata/JOB/utt2spk ark,s,cs:$cur_trans_dir/raw_trans.JOB ark:- ark:- | splice-feats $splice_opts ark:- ark:- |"
    feats="$splicedfeats transform-feats $dir/$cur_lda_iter.mat ark:- ark:- |"
  fi

  if echo $mllt_iters | grep -w $x >/dev/null; then
    if [ $stage -le $x ]; then
      echo "Estimating MLLT"
      $cmd JOB=1:$nj $dir/log/macc.$x.JOB.log \
        ali-to-post "ark:gunzip -c $dir/ali.JOB.gz|" ark:- \| \
        weight-silence-post 0.0 $silphonelist $dir/$x.mdl ark:- ark:- \| \
        gmm-acc-mllt --rand-prune=$randprune  $dir/$x.mdl "$feats" ark:- $dir/$x.JOB.macc \
        || exit 1;
      est-mllt $dir/$x.mat.new $dir/$x.*.macc 2> $dir/log/mupdate.$x.log || exit 1;
      gmm-transform-means  $dir/$x.mat.new $dir/$x.mdl $dir/$x.mdl \
        2> $dir/log/transform_means.$x.log || exit 1;
      compose-transforms --print-args=false $dir/$x.mat.new $dir/$cur_lda_iter.mat $dir/$x.mat || exit 1;
      rm $dir/$x.*.macc
    fi
    cur_lda_iter=$x
    feats="$splicedfeats transform-feats $dir/$cur_lda_iter.mat ark:- ark:- |"
  fi
  
  if [ $stage -le $x ]; then
    $cmd JOB=1:$nj $dir/log/acc.$x.JOB.log \
      gmm-acc-stats-ali $dir/$x.mdl "$feats" \
      "ark,s,cs:gunzip -c $dir/ali.JOB.gz|" $dir/$x.JOB.acc || exit 1;
    [ `ls $dir/$x.*.acc | wc -w` -ne "$nj" ] && echo "$0: Wrong #accs" && exit 1;
    $cmd $dir/log/update.$x.log \
      gmm-est --power=$power --write-occs=$dir/$[$x+1].occs --mix-up=$numgauss $dir/$x.mdl \
      "gmm-sum-accs - $dir/$x.*.acc |" $dir/$[$x+1].mdl || exit 1;
    rm $dir/$x.mdl $dir/$x.*.acc
    rm $dir/$x.occs 2>/dev/null
  fi
  [ $x -le $max_iter_inc ] && numgauss=$[$numgauss+$incgauss];
  x=$[$x+1];
done


if [ $stage -le $x ]; then
  # Accumulate stats for "alignment model"-- this model is
  # computed with the speaker-independent features, but matches Gaussian-for-Gaussian
  # with the final speaker-adapted model.
  sifeats="$sisplicedfeats transform-feats $dir/$cur_lda_iter.mat ark:- ark:- |"
  $cmd JOB=1:$nj $dir/log/acc_alimdl.JOB.log \
    ali-to-post "ark:gunzip -c $dir/ali.JOB.gz|" ark:-  \| \
    gmm-acc-stats-twofeats $dir/$x.mdl "$feats" "$sifeats" \
    ark,s,cs:- $dir/$x.JOB.acc || exit 1;
  [ `ls $dir/$x.*.acc | wc -w` -ne "$nj" ] && echo "$0: Wrong #accs" && exit 1;
  # Update model.
  $cmd $dir/log/est_alimdl.log \
    gmm-est --power=$power --remove-low-count-gaussians=false $dir/$x.mdl \
    "gmm-sum-accs - $dir/$x.*.acc|" $dir/$x.alimdl  || exit 1;
  rm $dir/$x.*.acc
fi

rm $dir/final.{mdl,alimdl,mat,occs} 2>/dev/null
ln -s $x.mdl $dir/final.mdl
ln -s $x.occs $dir/final.occs
ln -s $x.alimdl $dir/final.alimdl
ln -s $cur_lda_iter.mat $dir/final.mat


utils/summarize_warnings.pl $dir/log
(
  echo "$0: Likelihood evolution (not sure if this is totally correct):"
  for x in `seq $[$num_iters-1]`; do
    tail -n 30 $dir/log/acc.$x.*.log | awk '/Overall avg like/{l += $(NF-3)*$(NF-1); t += $(NF-1); }
        /Overall average logdet/{d += $(NF-3)*$(NF-1); t2 += $(NF-1);} 
        END{ d /= t2; l /= t; printf("%s ", d+l); } '
  done
  echo
) | tee $dir/log/summary.log

echo Done
