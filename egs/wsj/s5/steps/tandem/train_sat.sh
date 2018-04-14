#!/bin/bash
# Copyright 2012  Johns Hopkins University (Author: Daniel Povey).  Apache 2.0.
#                 Korbinian Riedhammer

# This does Speaker Adapted Training (SAT), i.e. train on
# fMLLR-adapted features.  It can be done on top of either LDA+MLLT, or
# delta and delta-delta features.  If there are no transforms supplied
# in the alignment directory, it will estimate transforms itself before
# building the tree (and in any case, it estimates transforms a number
# of times during training).


# Begin configuration section.
stage=-5
fmllr_update_type=full
cmd=run.pl
scale_opts="--transition-scale=1.0 --acoustic-scale=0.1 --self-loop-scale=0.1"
beam=10
retry_beam=40
boost_silence=1.0 # Factor by which to boost silence likelihoods in alignment
realign_iters="10 20 30";
fmllr_iters="2 4 6 12";
silence_weight=0.0 # Weight on silence in fMLLR estimation.
num_iters=35   # Number of iterations of training
max_iter_inc=25 # Last iter to increase #Gauss on.
power=0.2 # Exponent for number of gaussians according to occurrence counts
cluster_thresh=-1  # for build-tree control final bottom-up clustering of leaves
normft2=true
# End configuration section.

echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh
. parse_options.sh || exit 1;

if [ $# != 7 ]; then
  echo "Usage: steps/tandem/train_sat.sh <#leaves> <#gauss> <data1> <data2> <lang> <ali-dir> <exp-dir>"
  echo " e.g.: steps/tandem/train_sat.sh 2500 15000 {mfcc,bottleneck}/data/train_si84 data/lang exp/tri2b_ali_si84 exp/tri3b"
  echo "Main options (for others, see top of script file)"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --config <config-file>                           # config containing options"
  echo "  --stage <stage>                                  # stage to do partial re-run from."
  exit 1;
fi

numleaves=$1
totgauss=$2
data1=$3
data2=$4
lang=$5
alidir=$6
dir=$7

for f in $data1/feats.scp $data2/feats.scp $lang/phones.txt $alidir/final.mdl $alidir/ali.1.gz; do
  [ ! -f $f ] && echo "train_tandem_sat.sh: no such file $f" && exit 1;
done

numgauss=$numleaves
incgauss=$[($totgauss-$numgauss)/$max_iter_inc]  # per-iter #gauss increment
oov=`cat $lang/oov.int`
nj=`cat $alidir/num_jobs` || exit 1;
silphonelist=`cat $lang/phones/silence.csl`
ciphonelist=`cat $lang/phones/context_indep.csl` || exit 1;


mkdir -p $dir/log

utils/lang/check_phones_compatible.sh $lang/phones.txt $alidir/phones.txt || exit 1;
cp $lang/phones.txt $dir || exit 1;

echo $nj >$dir/num_jobs

sdata1=$data1/split$nj;
sdata2=$data2/split$nj;
[[ -d $sdata1 && $data1/feats.scp -ot $sdata1 ]] || split_data.sh $data1 $nj || exit 1;
[[ -d $sdata2 && $data2/feats.scp -ot $sdata2 ]] || split_data.sh $data2 $nj || exit 1;

# Set up features.

# We will use the same settings as with the alidir
splice_opts=`cat $alidir/splice_opts 2>/dev/null` # frame-splicing options.
normft2=`cat $alidir/normft2 2>/dev/null`

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
sifeats="ark,s,cs:paste-feats '$feats1' '$feats2' ark:- |"

# add transformation, if applicable
if [ "$feat_type" == "lda" ]; then
  sifeats="$sifeats transform-feats $dir/final.mat ark:- ark:- |"
fi

# splicing/normalization options
cp $alidir/{splice_opts,tandem,normft2} $dir 2>/dev/null



## Get initial fMLLR transforms (possibly from alignment dir)
if [ -f $alidir/trans.1 ]; then
  echo "$0: Using transforms from $alidir"
  feats="$sifeats transform-feats --utt2spk=ark:$sdata1/JOB/utt2spk ark,s,cs:$alidir/trans.JOB ark:- ark:- |"
  cur_trans_dir=$alidir
else
  if [ $stage -le -4 ]; then
    echo "$0: obtaining initial fMLLR transforms since not present in $alidir"
    $cmd JOB=1:$nj $dir/log/fmllr.0.JOB.log \
      ali-to-post "ark:gunzip -c $alidir/ali.JOB.gz|" ark:- \| \
      weight-silence-post $silence_weight $silphonelist $alidir/final.mdl ark:- ark:- \| \
      gmm-est-fmllr --fmllr-update-type=$fmllr_update_type \
      --spk2utt=ark:$sdata1/JOB/spk2utt $alidir/final.mdl "$sifeats" \
      ark:- ark:$dir/trans.JOB || exit 1;
  fi
  feats="$sifeats transform-feats --utt2spk=ark:$sdata1/JOB/utt2spk ark,s,cs:$dir/trans.JOB ark:- ark:- |"
  cur_trans_dir=$dir
fi

if [ $stage -le -3 ]; then
  # Get tree stats.
  echo "$0: Accumulating tree stats"
  $cmd JOB=1:$nj $dir/log/acc_tree.JOB.log \
    acc-tree-stats  --ci-phones=$ciphonelist $alidir/final.mdl "$feats" \
    "ark:gunzip -c $alidir/ali.JOB.gz|" $dir/JOB.treeacc || exit 1;
  [ "`ls $dir/*.treeacc | wc -w`" -ne "$nj" ] && echo "$0: Wrong #tree-accs" && exit 1;
  $cmd $dir/log/sum_tree_acc.log \
    sum-tree-stats $dir/treeacc $dir/*.treeacc || exit 1;
  rm $dir/*.treeacc
fi

if [ $stage -le -2 ]; then
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

  gmm-init-model  --write-occs=$dir/1.occs  \
    $dir/tree $dir/treeacc $lang/topo $dir/1.mdl 2> $dir/log/init_model.log || exit 1;
  grep 'no stats' $dir/log/init_model.log && echo "$0: This is a bad warning.";

  rm $dir/treeacc
fi


if [ $stage -le -1 ]; then
  # Convert the alignments.
  echo "$0: Converting alignments from $alidir to use current tree"
  $cmd JOB=1:$nj $dir/log/convert.JOB.log \
    convert-ali $alidir/final.mdl $dir/1.mdl $dir/tree \
     "ark:gunzip -c $alidir/ali.JOB.gz|" "ark:|gzip -c >$dir/ali.JOB.gz" || exit 1;
fi

if [ $stage -le 0 ]; then
  echo "$0: Compiling graphs of transcripts"
  $cmd JOB=1:$nj $dir/log/compile_graphs.JOB.log \
    compile-train-graphs --read-disambig-syms=$lang/phones/disambig.int $dir/tree $dir/1.mdl  $lang/L.fst  \
     "ark:utils/sym2int.pl --map-oov $oov -f 2- $lang/words.txt < $sdata1/JOB/text |" \
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
      $cmd JOB=1:$nj $dir/log/fmllr.$x.JOB.log \
        ali-to-post "ark:gunzip -c $dir/ali.JOB.gz|" ark:-  \| \
        weight-silence-post $silence_weight $silphonelist $dir/$x.mdl ark:- ark:- \| \
        gmm-est-fmllr --fmllr-update-type=$fmllr_update_type \
        --spk2utt=ark:$sdata1/JOB/spk2utt $dir/$x.mdl \
        "$feats" ark:- ark:$dir/tmp_trans.JOB || exit 1;
      for n in `seq $nj`; do
        ! ( compose-transforms --b-is-affine=true \
          ark:$dir/tmp_trans.$n ark:$cur_trans_dir/trans.$n ark:$dir/composed_trans.$n \
          && mv $dir/composed_trans.$n $dir/trans.$n && \
          rm $dir/tmp_trans.$n ) 2>$dir/log/compose_transforms.$x.log \
          && echo "$0: Error composing transforms" && exit 1;
      done
    fi
    feats="$sifeats transform-feats --utt2spk=ark:$sdata1/JOB/utt2spk ark:$dir/trans.JOB ark:- ark:- |"
    cur_trans_dir=$dir
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
    rm $dir/$x.occs
  fi
  [ $x -le $max_iter_inc ] && numgauss=$[$numgauss+$incgauss];
  x=$[$x+1];
done


if [ $stage -le $x ]; then
  # Accumulate stats for "alignment model"-- this model is
  # computed with the speaker-independent features, but matches Gaussian-for-Gaussian
  # with the final speaker-adapted model.
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

rm $dir/final.{mdl,alimdl,occs} 2>/dev/null
ln -s $x.mdl $dir/final.mdl
ln -s $x.occs $dir/final.occs
ln -s $x.alimdl $dir/final.alimdl



utils/summarize_warnings.pl $dir/log
(
  echo "$0: Likelihood evolution:"
  for x in `seq $[$num_iters-1]`; do
    tail -n 30 $dir/log/acc.$x.*.log | awk '/Overall avg like/{l += $(NF-3)*$(NF-1); t += $(NF-1); }
        /Overall average logdet/{d += $(NF-3)*$(NF-1); t2 += $(NF-1);}
        END{ d /= t2; l /= t; printf("%s ", d+l); } '
  done
  echo
) | tee $dir/log/summary.log

echo Done
