#!/bin/bash

# Copyright 2012-2013  Johns Hopkins University (Author: Daniel Povey),
#                      Idiap Research Institute (Author: David Imseng)
# Apache 2.0

# Begin configuration.
stage=-4 #  This allows restarting after partway, when something when wrong.
config=
cmd=run.pl
scale_opts="--transition-scale=1.0 --acoustic-scale=0.1 --self-loop-scale=0.1"
num_iters=35    # Number of iterations of training
max_iter_inc=25 # Last iter to increase #Gauss on.
beam=10
retry_beam=40
boost_silence=1.0 # Factor by which to boost silence likelihoods in alignment
power=0.25 # Exponent for number of gaussians according to occurrence counts
cluster_thresh=-1  # for build-tree control final bottom-up clustering of leaves
thresh=20
use_gpu="no"
nnet_dir=
context_opts=  # e.g. set this to "--context-width 5 --central-position 2" for quinphone.
tmpdir=
no_softmax=true
# End configuration.

echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh;
. parse_options.sh || exit 1;

if [ $# != 5 ]; then
   echo "Usage: steps/train_deltas.sh <num-leaves> <data-dir> <lang-dir> <alignment-dir> <exp-dir>"
   echo "e.g.: steps/train_deltas.sh 2000 data/train_si84_half data/lang exp/mono_ali exp/tri1"
   echo "main options (for others, see top of script file)"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   echo "  --config <config-file>                           # config containing options"
   echo "  --stage <stage>                                  # stage to do partial re-run from."
   echo "  --thresh   "
   echo "  --cluster_thresh "
   echo "  --nnet_dir "
   echo "  --context_opts "
   echo "  --tmpdir    "
   echo "  --no-softmax    "
   exit 1;
fi

numleaves=$1
data=$2
lang=$3
alidir=$4
dir=$5


for f in $alidir/final.mdl $alidir/ali.1.gz $data/feats.scp $lang/phones.txt; do
  [ ! -f $f ] && echo "train_deltas.sh: no such file $f" && exit 1;
done

numgauss=$numleaves
incgauss=$[($totgauss-$numgauss)/$max_iter_inc] # per-iter increment for #Gauss
ciphonelist=`cat $lang/phones/context_indep.csl` || exit 1;
nj=`cat $alidir/num_jobs` || exit 1;
mkdir -p $dir/log
echo $nj > $dir/num_jobs

sdata=$data/split$nj;
[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1;

nnet=${nnet_dir}/final.nnet
feature_transform=${nnet_dir}/final.feature_transform

featsdim="ark:copy-feats scp:$data/feats.scp ark:- |"
nnetfeats="ark,s,cs:copy-feats scp:$sdata/JOB/feats.scp ark:- |"
# Optionally add cmvn
if [ -f ${nnet_dir}/norm_vars ]; then
  norm_vars=$(cat ${nnet_dir}/norm_vars 2>/dev/null)
  [ ! -f $sdata/1/cmvn.scp ] && echo "$0: cannot find cmvn stats $sdata/1/cmvn.scp" && exit 1
  nnetfeats="$nnetfeats apply-cmvn --norm-vars=$norm_vars --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp ark:- ark:- |"
  featsdim="$featsdim apply-cmvn --norm-vars=$norm_vars --utt2spk=ark:$data/utt2spk scp:$data/cmvn.scp ark:- ark:- |"
fi
# Optionally add deltas
if [ -f ${nnet_dir}/delta_order ]; then
  delta_order=$(cat ${nnet_dir}/delta_order)
  nnetfeats="$nnetfeats add-deltas --delta-order=$delta_order ark:- ark:- |"
  featsdim="$featsdim add-deltas --delta-order=$delta_order ark:- ark:- |"
fi

feats="ark,s,cs:nnet-forward "
if [[ ! -z $feature_transform ]]; then 
    feats=${feats}" --feature-transform=$feature_transform "
fi
feats=${feats}"--no-softmax=$no_softmax --use-gpu=$use_gpu $nnet \"$nnetfeats\" ark:- |" 

feat_dim=$(feat-to-dim --print-args=false "$featsdim" -)
rm $dir/.error 2>/dev/null

if [[ ! -z $tmpdir ]]; then 
    mkdir -p $tmpdir 
else
    tmpdir=$dir
fi

if [ $stage -le -3 ]; then
  echo "$0: accumulating tree stats"
  $cmd JOB=1:$nj $dir/log/acc_tree.JOB.log \
    acc-tree-stats $context_opts --var-floor=1.0 --ci-phones=$ciphonelist $alidir/final.mdl "$feats" \
     "ark:gunzip -c $alidir/ali.JOB.gz|" $tmpdir/JOB.treeacc || exit 1;
  sum-tree-stats $dir/treeacc $tmpdir/*.treeacc 2>$dir/log/sum_tree_acc.log || exit 1;
  rm $tmpdir/*.treeacc
fi

if [ $stage -le -2 ]; then
  echo "$0: getting questions for tree-building, via clustering"
  # preparing questions, roots file...
  cluster-phones $context_opts $dir/treeacc $lang/phones/sets.int $dir/questions.int 2> $dir/log/questions.log || exit 1;
  cat $lang/phones/extra_questions.int >> $dir/questions.int
  compile-questions $context_opts $lang/topo $dir/questions.int $dir/questions.qst 2>$dir/log/compile_questions.log || exit 1;

  echo "$0: building the tree"
 # $cmd $dir/log/build_tree.log \
    build-tree $context_opts --verbose=1 --max-leaves=$numleaves --cluster-thresh=$cluster_thresh --thresh=$thresh $dir/treeacc $lang/phones/roots.int \
    $dir/questions.qst $lang/topo $dir/tree &> $dir/log/build_tree.log || exit 1;

  gmm-init-model-flat --dim=$feat_dim $dir/tree $lang/topo $dir/1.mdl

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
    compile-train-graphs $dir/tree $dir/1.mdl  $lang/L.fst  \
     "ark:utils/sym2int.pl -f 2- $lang/words.txt < $data/split$nj/JOB/text |" \
      "ark:|gzip -c >$dir/fsts.JOB.gz" || exit 1;
fi

rm $dir/final.mdl 2>/dev/null
ln -s 1.mdl $dir/final.mdl

# Summarize warning messages...
utils/summarize_warnings.pl  $dir/log

echo "$0: Done building the tree in $dir"

