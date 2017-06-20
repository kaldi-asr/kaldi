#!/bin/bash

# Copyright 2012  Johns Hopkins University (Author: Daniel Povey)
#                 Korbinian Riedhammer
# Apache 2.0.

# This is a vanilla tandem system where the first stream is just extended with
# delta+deltadeltas, in contrast to the train_lda_mllt.sh script, where the
# temoporal context of the first stream is modeled via HLDA

# Begin configuration.
cmd=run.pl
config=
stage=-5
scale_opts="--transition-scale=1.0 --acoustic-scale=0.1 --self-loop-scale=0.1"
realign_iters="10 20 30";
mllt_iters="2 4 6 12";
num_iters=35    # Number of iterations of training
max_iter_inc=25  # Last iter to increase #Gauss on.
beam=10
retry_beam=40
boost_silence=1.0 # Factor by which to boost silence likelihoods in alignment
power=0.2 # Exponent for number of gaussians according to occurrence counts
randprune=4.0 # This is approximately the ratio by which we will speed up the
              # LDA and MLLT calculations via randomized pruning.
cluster_thresh=-1  # for build-tree control final bottom-up clustering of leaves

# apply CMVN to the second feature stream?
normft2=true

# Do additional LDA after pasting the features
dim2=40
extra_lda=false

# End configuration.

echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh
. parse_options.sh || exit 1;

if [ $# != 7 ]; then
  echo "Usage: steps/tandem/train_mllt.sh [options] <#leaves> <#gauss> <data1> <data2> <lang> <alignments> <dir>"
  echo " e.g.: steps/tandem/train_mllt.sh 2500 15000 {mfcc,bottleneck}/data/train_si84 data/lang exp/tri1_ali_si84 exp/tri2b"
  echo "Main options (for others, see top of script file)"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --config <config-file>                           # config containing options"
  echo "  --stage <stage>                                  # stage to do partial re-run from."
  echo "  --normft2 (true|false)                           # apply CMVN to second data set (true)"
  echo "  --extra-lda (true|false)                         # apply extra LDA after feature paste (false)"
  echo "  --dim2 <n>                                       # dimension of the pasted features after 2nd HLDA"
  exit 1;
fi

numleaves=$1
totgauss=$2
data1=$3
data2=$4
lang=$5
alidir=$6
dir=$7

for f in $alidir/final.mdl $alidir/ali.1.gz $data1/feats.scp $data2/feats.scp $lang/phones.txt; do
  [ ! -f $f ] && echo "train_tandem_lda_mllt.sh: no such file $f" && exit 1;
done

numgauss=$numleaves
incgauss=$[($totgauss-$numgauss)/$max_iter_inc] # per-iter #gauss increment
oov=`cat $lang/oov.int` || exit 1;
nj=`cat $alidir/num_jobs` || exit 1;
silphonelist=`cat $lang/phones/silence.csl` || exit 1;
ciphonelist=`cat $lang/phones/context_indep.csl` || exit 1;

mkdir -p $dir/log
echo $nj >$dir/num_jobs

utils/lang/check_phones_compatible.sh $lang/phones.txt $alidir/phones.txt || exit 1;
cp $lang/phones.txt $dir || exit 1;


# Set up features.

sdata1=$data1/split$nj;
sdata2=$data2/split$nj;
[[ -d $sdata1 && $data1/feats.scp -ot $sdata1 ]] || split_data.sh $data1 $nj || exit 1;
[[ -d $sdata2 && $data2/feats.scp -ot $sdata2 ]] || split_data.sh $data2 $nj || exit 1;

# set up feature stream 1;  here we assume spectral features which we will 
# splice instead of deltas
feats1="ark,s,cs:apply-cmvn --norm-vars=false --utt2spk=ark:$sdata1/JOB/utt2spk scp:$sdata1/JOB/cmvn.scp scp:$sdata1/JOB/feats.scp ark:- | add-deltas ark:- ark:- |"

# set up feature stream 2;  this are usually bottleneck or posterior features, 
# which may be normalized if desired
feats2="scp:$sdata2/JOB/feats.scp"

if [ "$normft2" == "true" ]; then
  feats2="ark,s,cs:apply-cmvn --norm-vars=false --utt2spk=ark:$sdata2/JOB/utt2spk scp:$sdata2/JOB/cmvn.scp $feats2 ark:- |"
fi

# assemble tandem features;  note: $feats gets overwritten later in the script
# once we have MLLT matrices
tandemfeats="ark,s,cs:paste-feats '$feats1' '$feats2' ark:- |"
feats="$tandemfeats"

# keep track of splicing/normalization options
echo $feats > $dir/tandem
echo $normft2 > $dir/normft2


# Begin training;  initially, we have no MLLT matrix
cur_mllt_iter=0

if [ $stage -le -4 -a $extra_lda == true ]; then
  echo "Accumulating LDA statistics (for tandem features this time)."
  $cmd JOB=1:$nj $dir/log/lda_acc.JOB.log \
    ali-to-post "ark:gunzip -c $alidir/ali.JOB.gz|" ark:- \| \
    weight-silence-post 0.0 $silphonelist $alidir/final.mdl ark:- ark:- \| \
    acc-lda --rand-prune=$randprune $alidir/final.mdl "$tandemfeats" ark,s,cs:- \
    $dir/lda.JOB.acc || exit 1;
  est-lda --write-full-matrix=$dir/full.mat --dim=$dim2 $dir/0.mat $dir/lda.*.acc \
    2>$dir/log/lda_est.log || exit 1;
  rm $dir/lda.*.acc
  
  feats="$tandemfeats transform-feats $dir/0.mat ark:- ark:- |"
fi

if [ $stage -le -3 ]; then
  echo "Accumulating tree stats"
  $cmd JOB=1:$nj $dir/log/acc_tree.JOB.log \
   acc-tree-stats  --ci-phones=$ciphonelist $alidir/final.mdl "$feats" \
     "ark:gunzip -c $alidir/ali.JOB.gz|" $dir/JOB.treeacc || exit 1;
  [ `ls $dir/*.treeacc | wc -w` -ne "$nj" ] && echo "Wrong #tree-accs" && exit 1;
  $cmd $dir/log/sum_tree_acc.log \
    sum-tree-stats $dir/treeacc $dir/*.treeacc || exit 1;
  rm $dir/*.treeacc
fi


if [ $stage -le -2 ]; then
  echo "Getting questions for tree clustering."
  # preparing questions, roots file...
  cluster-phones $dir/treeacc $lang/phones/sets.int $dir/questions.int 2> $dir/log/questions.log || exit 1;
  cat $lang/phones/extra_questions.int >> $dir/questions.int
  compile-questions $lang/topo $dir/questions.int $dir/questions.qst 2>$dir/log/compile_questions.log || exit 1;

  echo "Building the tree"
  $cmd $dir/log/build_tree.log \
    build-tree --verbose=1 --max-leaves=$numleaves \
    --cluster-thresh=$cluster_thresh $dir/treeacc $lang/phones/roots.int \
    $dir/questions.qst $lang/topo $dir/tree || exit 1;

  gmm-init-model  --write-occs=$dir/1.occs  \
    $dir/tree $dir/treeacc $lang/topo $dir/1.mdl 2> $dir/log/init_model.log || exit 1;
  grep 'no stats' $dir/log/init_model.log && echo "This is a bad warning.";

  # could mix up if we wanted:
  # gmm-mixup --mix-up=$numgauss $dir/1.mdl $dir/1.occs $dir/1.mdl 2>$dir/log/mixup.log || exit 1;
  rm $dir/treeacc
fi


if [ $stage -le -1 ]; then
  # Convert the alignments.
  echo "Converting alignments from $alidir to use current tree"
  $cmd JOB=1:$nj $dir/log/convert.JOB.log \
    convert-ali $alidir/final.mdl $dir/1.mdl $dir/tree \
     "ark:gunzip -c $alidir/ali.JOB.gz|" "ark:|gzip -c >$dir/ali.JOB.gz" || exit 1;
fi

if [ $stage -le 0 ]; then
  echo "Compiling graphs of transcripts"
  $cmd JOB=1:$nj $dir/log/compile_graphs.JOB.log \
    compile-train-graphs --read-disambig-syms=$lang/phones/disambig.int $dir/tree $dir/1.mdl  $lang/L.fst  \
     "ark:utils/sym2int.pl --map-oov $oov -f 2- $lang/words.txt < $data1/split$nj/JOB/text |" \
      "ark:|gzip -c >$dir/fsts.JOB.gz" || exit 1;
fi


x=1
while [ $x -lt $num_iters ]; do
  echo Training pass $x
  if echo $realign_iters | grep -w $x >/dev/null && [ $stage -le $x ]; then
    echo Aligning data
    mdl="gmm-boost-silence --boost=$boost_silence `cat $lang/phones/optional_silence.csl` $dir/$x.mdl - |"
    $cmd JOB=1:$nj $dir/log/align.$x.JOB.log \
      gmm-align-compiled $scale_opts --beam=$beam --retry-beam=$retry_beam "$mdl" \
      "ark:gunzip -c $dir/fsts.JOB.gz|" "$feats" \
      "ark:|gzip -c >$dir/ali.JOB.gz" || exit 1;
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
      
      # see if this is the first MLLT iteration and there is no lda;  otherwise compose transforms
      if [ $cur_mllt_iter == 0 -a $extra_lda == false ]; then
        mv $dir/$x.mat.new $dir/$x.mat || exit 1;
      else
        compose-transforms --print-args=false $dir/$x.mat.new $dir/$cur_mllt_iter.mat $dir/$x.mat || exit 1;
      fi

      rm $dir/$x.*.macc
    fi

    # update features
    feats="$tandemfeats transform-feats $dir/$x.mat ark:- ark:- |"
    cur_mllt_iter=$x
  fi

  if [ $stage -le $x ]; then
    $cmd JOB=1:$nj $dir/log/acc.$x.JOB.log \
      gmm-acc-stats-ali  $dir/$x.mdl "$feats" \
      "ark,s,cs:gunzip -c $dir/ali.JOB.gz|" $dir/$x.JOB.acc || exit 1;
    $cmd $dir/log/update.$x.log \
      gmm-est --write-occs=$dir/$[$x+1].occs --mix-up=$numgauss --power=$power \
        $dir/$x.mdl "gmm-sum-accs - $dir/$x.*.acc |" $dir/$[$x+1].mdl || exit 1;
    rm $dir/$x.mdl $dir/$x.*.acc $dir/$x.occs 
  fi
  [ $x -le $max_iter_inc ] && numgauss=$[$numgauss+$incgauss];
  x=$[$x+1];
done

rm $dir/final.{mdl,mat,occs} 2>/dev/null
ln -s $x.mdl $dir/final.mdl
ln -s $x.occs $dir/final.occs
ln -s $cur_mllt_iter.mat $dir/final.mat

# Summarize warning messages...

utils/summarize_warnings.pl $dir/log

echo Done training system with LDA+MLLT tandem features in $dir
