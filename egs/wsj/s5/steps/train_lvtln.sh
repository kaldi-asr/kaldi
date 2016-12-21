#!/bin/bash

# Copyright 2012-2014  Johns Hopkins University (Author: Daniel Povey)
# Copyright 2014       Vimal Manohar
# This training script trains linear-VTLN models starting from an existing
# system based on either LDA+MLLT or delta+delta-delta features.
# Works with either mfcc or plp features, but you need to set the 
# --base-feat-type option.
# The resulting system can be used with align_lvtln.sh and/or decode_lvtln.sh
# to get VTLN warping factors for data, for warped data extraction, or (for
# the training data) you can use the warping factors this script outputs
# in $dir/final.warp
#
# Apache 2.0

# Begin configuration.
stage=-6 #  This allows restarting after partway, when something when wrong.
config=
cmd=run.pl
scale_opts="--transition-scale=1.0 --acoustic-scale=0.1 --self-loop-scale=0.1"
realign_iters="10 20 30";
num_iters=35    # Number of iterations of training
max_iter_inc=25 # Last iter to increase #Gauss on.
beam=10
retry_beam=40
boost_silence=1.0 # Factor by which to boost silence likelihoods in alignment
power=0.25 # Exponent for number of gaussians according to occurrence counts
cluster_thresh=-1  # for build-tree control final bottom-up clustering of leaves
cmvn_opts=  # you can supply e.g. --cmvn-opts "--norm-vars=true" to turn on variance
            # normalization, but only if base system is the delta type, not LDA.
lvtln_iters="2 4 6 8 10 12 14 16 20"; # iters on which to recompute LVTLN transform"
num_utt_lvtln_init=200; # number of utterances (subset) to initialize
                        # LVTLN transform.  Not too critical.
min_warp=0.85
max_warp=1.25
warp_step=0.01
base_feat_type=mfcc # or could be PLP.
logdet_scale=0.0

# End configuration.

echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh;
. parse_options.sh || exit 1;

num_classes=$(perl -e "print int(1.5 + ($max_warp - $min_warp) / $warp_step);") || exit 1;
default_class=$(perl -e "print int(0.5 + (1.0 - $min_warp) / $warp_step);") || exit 1;

if [ $# != 6 ]; then
   echo "Usage: $0 <num-leaves> <tot-gauss> <data-dir> <lang-dir> <alignment-dir> <exp-dir>"
   echo "e.g.: $0 2000 10000 data/train_si84_half data/lang exp/mono_ali exp/tri1"
   echo "main options (for others, see top of script file)"
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

for f in $alidir/final.mdl $alidir/ali.1.gz $data/feats.scp $lang/phones.txt $data/wav.scp; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

numgauss=$numleaves
incgauss=$[($totgauss-$numgauss)/$max_iter_inc] # per-iter increment for #Gauss
oov=`cat $lang/oov.int` || exit 1;
silphonelist=`cat $lang/phones/silence.csl` || exit 1;
ciphonelist=`cat $lang/phones/context_indep.csl` || exit 1;
nj=`cat $alidir/num_jobs` || exit 1;
splice_opts=`cat $alidir/splice_opts 2>/dev/null`
mkdir -p $dir/log
echo $nj > $dir/num_jobs

utils/lang/check_phones_compatible.sh $lang/phones.txt $alidir/phones.txt || exit 1;
cp $lang/phones.txt $dir || exit 1;

sdata=$data/split$nj;
split_data.sh $data $nj || exit 1;


cp $alidir/splice_opts $dir 2>/dev/null


if [ ! -f $alidir/final.mat ]; then
  [ $(cat $alidir/cmvn_opts 2>/dev/null | wc -c) -gt 1 ] && [ -z "$cmvn_opts" ] && \
    echo "$0: warning: ignoring CMVN options from $alidir.";
  echo $cmvn_opts > $dir/cmvn_opts

  echo "$0: Using delta+delta-delta features since $alidir/final.mat does not exist"
  sifeats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | add-deltas ark:- ark:- |"
  feats="$sifeats transform-feats --utt2spk=ark:$sdata/JOB/utt2spk ark:$dir/trans.JOB ark:- ark:- |"
  # for the subsets of features that we use to estimate the linear transforms, we don't
  # bother with CMVN.  This will give us wrong offsets on the transforms, but it will end
  # up not mattering because we allow an arbitrary offset (bias) term when we apply
  # these transforms.
  featsub_warped="ark:add-deltas ark:$dir/feats.CLASS.ark ark:- |" # you need to define CLASS when invoking $cmd.
  featsub_unwarped="ark:add-deltas ark:$dir/feats.$default_class.ark ark:- |"
else
  echo "$0: Using LDA features"
  [ ! -z "$cmvn_opts" ] && echo  "$0: you cannot supply --cmvn-opts if base system is LDA."
  cp $alidir/final.mat $alidir/full.mat $alidir/splice_opts $alidir/cmvn_opts $dir 2>/dev/null 
  cmvn_opts=`cat $dir/cmvn_opts 2>/dev/null`
  sifeats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | splice-feats $splice_opts ark:- ark:- | transform-feats $dir/final.mat ark:- ark:- |"
  feats="$sifeats transform-feats --utt2spk=ark:$sdata/JOB/utt2spk ark:$dir/trans.JOB ark:- ark:- |"
  featsub_warped="ark:splice-feats $splice_opts ark:$dir/feats.CLASS.ark ark:- | transform-feats $dir/final.mat ark:- ark:- |" # you need to define CLASS when invoking $cmd.
  featsub_unwarped="ark:splice-feats $splice_opts ark:$dir/feats.$default_class.ark ark:- | transform-feats $dir/final.mat ark:- ark:- |"  
fi

if [ -f $data/utt2warp ]; then
  echo "$0: source data directory $data appears to already have VTLN.";
  exit 1;
fi

# create a small subset of utterances for purposes of initializing the LVTLN transform
# utils/shuffle_list.pl is deterministic, unlike sort -R.
cat $data/utt2spk | awk '{print $1}' | utils/shuffle_list.pl | \
  head -n $num_utt_lvtln_init > $dir/utt_subset

if [ $stage -le -6 ]; then
  echo "$0: computing warped subset of features"
  if [ -f $data/segments ]; then
    echo "$0 [info]: segments file exists: using that."
    subset_feats="utils/filter_scp.pl $dir/utt_subset $data/segments | extract-segments scp:$data/wav.scp - ark:- "
  else
    echo "$0 [info]: no segments file exists: using wav.scp directly."
    subset_feats="utils/filter_scp.pl $dir/utt_subset $data/wav.scp | wav-copy scp:- ark:- "
  fi
  rm $dir/.error 2>/dev/null
  for c in $(seq 0 $[$num_classes-1]); do
    this_warp=$(perl -e "print ($min_warp + ($c*$warp_step));")
    $cmd $dir/log/compute_warped_feats.$c.log \
      $subset_feats \| compute-${base_feat_type}-feats --verbose=2 \
      --config=conf/${base_feat_type}.conf --vtln-warp=$this_warp ark:- ark:- \| \
      copy-feats --compress=true ark:- ark:$dir/feats.$c.ark || touch $dir/.error &
  done
  wait;
  if [ -f $dir/.error ]; then
    echo "$0: Computing warped features failed: check $dir/log/compute_warped_feats.*.log"
    exit 1;
  fi
fi

if ! utils/filter_scp.pl $dir/utt_subset $data/feats.scp | \
  compare-feats --threshold=0.98 scp:-  ark:$dir/feats.$default_class.ark >&/dev/null; then
  echo "$0: features stored on disk differ from those computed with no warping."
  echo "    Possibly your feature type is wrong (--base-feat-type option)"
  exit 1;
fi
  
if [ -f $data/segments ]; then
  subset_utts="ark:extract-segments scp:$sdata/JOB/wav.scp $sdata/JOB/segments ark:- |"
else
  echo "$0 [info]: no segments file exists: using wav.scp directly."
  subset_utts="ark:wav-copy scp:$sdata/JOB/wav.scp ark:- |"
fi

if [ $stage -le -5 ]; then
  echo "$0: initializing base LVTLN transforms in $dir/0.lvtln (ignore warnings below)"
  dim=$(feat-to-dim "$featsub_unwarped" - ) || exit 1;

  $cmd $dir/log/init_lvtln.log \
    gmm-init-lvtln --dim=$dim --num-classes=$num_classes --default-class=$default_class \
      $dir/0.lvtln || exit 1;

  $cmd JOB=1:$nj $dir/log/get_weights.JOB.log \
    ali-to-post "ark:gunzip -c $alidir/ali.JOB.gz |" ark:- \| \
    weight-silence-post 0.0 "$silphonelist" $alidir/final.mdl ark:- ark:- \| \
    post-to-weights ark:- "ark,scp:$dir/weights.JOB.ark,$dir/weights.JOB.scp" || exit 1

  for n in `seq 1 $nj`; do 
    cat $dir/weights.$n.scp
  done > $dir/weights.scp

  for c in $(seq 0 $[$num_classes-1]); do
    this_warp=$(perl -e "print ($min_warp + ($c*$warp_step));")
    orig_feats=ark:$dir/feats.$default_class.ark
    warped_feats=ark:$dir/feats.$c.ark
    logfile=$dir/log/train_special.$c.log
    this_featsub_warped="$(echo $featsub_warped | sed s/CLASS/$c/)"
    if ! gmm-train-lvtln-special --warp=$this_warp --normalize-var=true \
      --weights-in="scp:$dir/weights.scp" \
      $c $dir/0.lvtln $dir/0.lvtln \
      "$featsub_unwarped" "$this_featsub_warped" 2>$logfile; then
      echo "$0: Error training LVTLN transform, see $logfile";
      exit 1;
    fi
  done  
  rm $dir/final.lvtln 2>/dev/null
  ln -s 0.lvtln $dir/final.lvtln
fi

if [ $stage -le -4 ]; then
  echo "$0: computing initial LVTLN transforms for speakers"

  if [ -f $alidir/final.alimdl ]; then
    # if the base system was trained with SAT, it's probably better
    # to use the .alimdl, trained speaker-independent, to get the
    # LVTLN transforms (LVTLN may be closer to an unadapted system).
    echo "$0: to get initial LVTLN transforms, using $alidir/final.alimdl"
    srcmodel=$alidir/final.alimdl
  else
    srcmodel=$alidir/final.mdl
  fi

  $cmd JOB=1:$nj $dir/log/lvtln.0.JOB.log \
    ali-to-post "ark:gunzip -c $alidir/ali.JOB.gz|" ark:- \| \
    weight-silence-post 0.0 "$silphonelist" $alidir/final.mdl ark:- ark:- \| \
    gmm-post-to-gpost $srcmodel "$sifeats" ark:- ark:- \| \
    gmm-est-lvtln-trans --logdet-scale=$logdet_scale --verbose=1 \
      --spk2utt=ark:$sdata/JOB/spk2utt $srcmodel \
      $dir/0.lvtln "$sifeats" ark:- ark:$dir/trans.JOB ark,t:$dir/warp.0.JOB || exit 1
  
  # consolidate the warps into one file.
  for j in $(seq $nj); do cat $dir/warp.0.$j; done > $dir/warp.0
  rm $dir/warp.0.*
fi

if [ $stage -le -3 ]; then
  echo "$0: accumulating tree stats"
  $cmd JOB=1:$nj $dir/log/acc_tree.JOB.log \
    acc-tree-stats  --ci-phones=$ciphonelist $alidir/final.mdl "$feats" \
     "ark:gunzip -c $alidir/ali.JOB.gz|" $dir/JOB.treeacc || exit 1;
  sum-tree-stats $dir/treeacc $dir/*.treeacc 2>$dir/log/sum_tree_acc.log || exit 1;
  rm $dir/*.treeacc
fi

if [ $stage -le -2 ]; then
  echo "$0: getting questions for tree-building, via clustering"
  # preparing questions, roots file...
  cluster-phones $dir/treeacc $lang/phones/sets.int $dir/questions.int 2> $dir/log/questions.log || exit 1;
  cat $lang/phones/extra_questions.int >> $dir/questions.int
  compile-questions $lang/topo $dir/questions.int $dir/questions.qst 2>$dir/log/compile_questions.log || exit 1;

  echo "$0: building the tree"
  $cmd $dir/log/build_tree.log \
    build-tree --verbose=1 --max-leaves=$numleaves \
    --cluster-thresh=$cluster_thresh $dir/treeacc $lang/phones/roots.int \
    $dir/questions.qst $lang/topo $dir/tree || exit 1;

  gmm-init-model  --write-occs=$dir/1.occs  \
    $dir/tree $dir/treeacc $lang/topo $dir/1.mdl 2> $dir/log/init_model.log || exit 1;
  grep 'no stats' $dir/log/init_model.log && echo "This is a bad warning.";

  gmm-mixup --mix-up=$numgauss $dir/1.mdl $dir/1.occs $dir/1.mdl 2>$dir/log/mixup.log || exit 1;
  rm $dir/treeacc
fi

if [ $stage -le -1 ]; then
  # Convert the alignments.
  echo "$0: converting alignments from $alidir to use current tree"
  $cmd JOB=1:$nj $dir/log/convert.JOB.log \
    convert-ali $alidir/final.mdl $dir/1.mdl $dir/tree \
     "ark:gunzip -c $alidir/ali.JOB.gz|" "ark:|gzip -c >$dir/ali.JOB.gz" || exit 1;
fi

if [ $stage -le 0 ]; then
  echo "$0: compiling graphs of transcripts"
  $cmd JOB=1:$nj $dir/log/compile_graphs.JOB.log \
    compile-train-graphs --read-disambig-syms=$lang/phones/disambig.int $dir/tree $dir/1.mdl  $lang/L.fst  \
     "ark:utils/sym2int.pl --map-oov $oov -f 2- $lang/words.txt < $data/split$nj/JOB/text |" \
      "ark:|gzip -c >$dir/fsts.JOB.gz" || exit 1;
fi

x=1
while [ $x -lt $num_iters ]; do
  echo "$0: training pass $x"
  if echo $realign_iters | grep -w $x >/dev/null; then
    if [ $stage -le $x ]; then
      echo "$0: aligning data"
      mdl="gmm-boost-silence --boost=$boost_silence `cat $lang/phones/optional_silence.csl` $dir/$x.mdl - |"
      $cmd JOB=1:$nj $dir/log/align.$x.JOB.log \
        gmm-align-compiled $scale_opts --beam=$beam --retry-beam=$retry_beam "$mdl" \
         "ark:gunzip -c $dir/fsts.JOB.gz|" "$feats" \
         "ark:|gzip -c >$dir/ali.JOB.gz" || exit 1;
    fi
  fi
  if echo $lvtln_iters | grep -w $x >/dev/null; then
    if [ $stage -le $x ]; then
      echo "Re-estimating LVTLN transforms"
      $cmd JOB=1:$nj $dir/log/lvtln.$x.JOB.log \
        ali-to-post "ark:gunzip -c $dir/ali.JOB.gz|" ark:-  \| \
        weight-silence-post 0.0 $silphonelist $dir/$x.mdl ark:- ark:- \| \
        gmm-post-to-gpost $dir/$x.mdl "$feats" ark:- ark:- \| \
        gmm-est-lvtln-trans --logdet-scale=$logdet_scale --verbose=1 \
          --spk2utt=ark:$sdata/JOB/spk2utt $dir/$x.mdl \
          $dir/0.lvtln "$sifeats" ark:- ark:$dir/new_trans.JOB ark,t:$dir/warp.$x.JOB || exit 1
      # consolidate the warps into one file.
      for j in $(seq $nj); do mv $dir/new_trans.$j $dir/trans.$j; done
      for j in $(seq $nj); do cat $dir/warp.$x.$j; done > $dir/warp.$x
      rm $dir/warp.$x.*
    fi
  fi

  if [ $stage -le $x ]; then
    $cmd JOB=1:$nj $dir/log/acc.$x.JOB.log \
      gmm-acc-stats-ali  $dir/$x.mdl "$feats" \
      "ark,s,cs:gunzip -c $dir/ali.JOB.gz|" $dir/$x.JOB.acc || exit 1;
    $cmd $dir/log/update.$x.log \
      gmm-est --mix-up=$numgauss --power=$power \
      --write-occs=$dir/$[$x+1].occs $dir/$x.mdl \
      "gmm-sum-accs - $dir/$x.*.acc |" $dir/$[$x+1].mdl || exit 1;
    rm $dir/$x.mdl $dir/$x.*.acc
    rm $dir/$x.occs
  fi
  [ $x -le $max_iter_inc ] && numgauss=$[$numgauss+$incgauss];
  x=$[$x+1];
done


if [ $stage -le $x ]; then
  # Accumulate stats for "alignment model"-- this model is computed with the
  # speaker-independent features, but matches Gaussian-for-Gaussian with the
  # final speaker-adapted model.
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

if true; then # Diagnostics
  last_iter=$(echo 0 $lvtln_iters  | awk '{print $NF;}')
  ln -sf warp.$last_iter $dir/final.warp
  if [ -f $data/spk2gender ]; then 
    # To make it easier to eyeball the male and female speakers' warps
    # separately, separate them out.
    for g in m f; do # means: for gender in male female
      cat $dir/final.warp | \
        utils/filter_scp.pl <(grep -w $g $data/spk2gender | awk '{print $1}') > $dir/final.warp.$g
      echo -n "The last few warp factors for gender $g are: "
      tail -n 10 $dir/final.warp.$g | awk '{printf("%s ", $2);}'; 
      echo
    done
  fi
fi

ln -sf $x.mdl $dir/final.mdl
ln -sf $x.occs $dir/final.occs
ln -sf $x.alimdl $dir/final.alimdl

# Summarize warning messages...
utils/summarize_warnings.pl  $dir/log

echo "$0: Done training LVTLN system in $dir"
