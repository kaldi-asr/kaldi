#!/bin/bash
# Copyright 2012  Johns Hopkins University (Author: Daniel Povey)
#                 Korbinian Riedhammer
# Apache 2.0


# To be run from ..
# Flat start and monophone training, with delta-delta features.
# This script applies cepstral mean normalization (per speaker).

# Begin configuration section.
nj=4
cmd=run.pl
scale_opts="--transition-scale=1.0 --acoustic-scale=0.1 --self-loop-scale=0.1"
num_iters=40    # Number of iterations of training
max_iter_inc=30 # Last iter to increase #Gauss on.
totgauss=1000 # Target #Gaussians.  
boost_silence=1.0 # Factor by which to boost silence likelihoods in alignment
realign_iters="1 2 3 4 5 6 7 8 9 10 12 14 16 18 20 23 26 29 32 35 38";
config= # name of config file.
stage=-4
power=0.2 # exponent to determine number of gaussians from occurrence counts
normft2=true # typically, the tandem features will already be normalized due to pca
# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 4 ]; then
  echo "Usage: steps/tandem/train_mono.sh [options] <data1-dir> <data2-dir> <lang-dir> <exp-dir>"
  echo " e.g.: steps/tandem/train_mono.sh {mfcc,bottleneck}/data/train.1k data/lang exp/mono"
  echo "main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config containing options"
  echo "  --nj <nj>                                        # number of parallel jobs"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --normft2 (true|false)                           # apply CMVN to second features?"
  exit 1;
fi

data1=$1
data2=$2
lang=$3
dir=$4

oov_sym=`cat $lang/oov.int` || exit 1;

mkdir -p $dir/log
echo $nj > $dir/num_jobs


# Set up features.

sdata1=$data1/split$nj;
sdata2=$data2/split$nj;
[[ -d $sdata1 && $data1/feats.scp -ot $sdata1 ]] || split_data.sh $data1 $nj || exit 1;
[[ -d $sdata2 && $data2/feats.scp -ot $sdata2 ]] || split_data.sh $data2 $nj || exit 1;

# Use deltas on the first tream (most likely this will be MFCCs or alike)
feats1="ark,s,cs:apply-cmvn --norm-vars=false --utt2spk=ark:$sdata1/JOB/utt2spk scp:$sdata1/JOB/cmvn.scp scp:$sdata1/JOB/feats.scp ark:- | add-deltas ark:- ark:- |"

# Second stream will most likely be bottleneck or posteriors, so normalize
# if desired
feats2="scp:$sdata2/JOB/feats.scp"
if [ "$normft2" == "true" ]; then
  feats2="ark,s,cs:apply-cmvn --norm-vars=false --utt2spk=ark:$sdata2/JOB/utt2spk scp:$sdata2/JOB/cmvn.scp $feats2 ark:- |"
fi

# paste features
feats="ark,s,cs:paste-feats '$feats1' '$feats2' ark:- |"
example_feats="`echo '$feats' | sed s/JOB/1/g`";

# get dimension
allfeats=$(echo $feats | sed s:JOB:..:g)
feat_dim=$(feat-to-dim --print-args=false "$allfeats" - 2> $dir/log/feat_dim)

# save stats
echo $feats > $dir/tandem
echo $normft2 > $dir/normft2

echo "$0: Initializing monophone system."

[ ! -f $lang/phones/sets.int ] && exit 1;
shared_phones_opt="--shared-phones=$lang/phones/sets.int"

if [ $stage -le -3 ]; then
# Note: JOB=. makes it use the whole set;  we want that to make sure we have phoneme 
  $cmd JOB=1 $dir/log/init.log \
    gmm-init-mono $shared_phones_opt "--train-feats=$allfeats" $lang/topo $feat_dim \
    $dir/0.mdl $dir/tree || exit 1;
fi

numgauss=`gmm-info --print-args=false $dir/0.mdl | grep gaussians | awk '{print $NF}'`
incgauss=$[($totgauss-$numgauss)/$max_iter_inc] # per-iter increment for #Gauss

if [ $stage -le -2 ]; then
  echo "$0: Compiling training graphs"
  $cmd JOB=1:$nj $dir/log/compile_graphs.JOB.log \
    compile-train-graphs $dir/tree $dir/0.mdl  $lang/L.fst \
    "ark:sym2int.pl --map-oov $oov_sym -f 2- $lang/words.txt < $sdata1/JOB/text|" \
    "ark:|gzip -c >$dir/fsts.JOB.gz" || exit 1;
fi

if [ $stage -le -1 ]; then
  echo "$0: Aligning data equally (pass 0)"
  $cmd JOB=1:$nj $dir/log/align.0.JOB.log \
    align-equal-compiled "ark:gunzip -c $dir/fsts.JOB.gz|" "$feats" ark,t:-  \| \
    gmm-acc-stats-ali --binary=true $dir/0.mdl "$feats" ark:- \
    $dir/0.JOB.acc || exit 1;
fi

# In the following steps, the --min-gaussian-occupancy=3 option is important, otherwise
# we fail to est "rare" phones and later on, they never align properly.

if [ $stage -le 0 ]; then
  gmm-est --min-gaussian-occupancy=3  --mix-up=$numgauss --power=$power \
    $dir/0.mdl "gmm-sum-accs - $dir/0.*.acc|" $dir/1.mdl 2> $dir/log/update.0.log || exit 1;
  rm $dir/0.*.acc
fi


beam=6 # will change to 10 below after 1st pass
# note: using slightly wider beams for WSJ vs. RM.
x=1
while [ $x -lt $num_iters ]; do
  echo "$0: Pass $x"
  if [ $stage -le $x ]; then
    if echo $realign_iters | grep -w $x >/dev/null; then
      echo "$0: Aligning data"
      mdl="gmm-boost-silence --boost=$boost_silence `cat $lang/phones/optional_silence.csl` $dir/$x.mdl - |"
      $cmd JOB=1:$nj $dir/log/align.$x.JOB.log \
        gmm-align-compiled $scale_opts --beam=$beam --retry-beam=$[$beam*4] "$mdl" \
        "ark:gunzip -c $dir/fsts.JOB.gz|" "$feats" "ark,t:|gzip -c >$dir/ali.JOB.gz" \
        || exit 1;
    fi
    $cmd JOB=1:$nj $dir/log/acc.$x.JOB.log \
      gmm-acc-stats-ali  $dir/$x.mdl "$feats" "ark:gunzip -c $dir/ali.JOB.gz|" \
      $dir/$x.JOB.acc || exit 1;

    $cmd $dir/log/update.$x.log \
      gmm-est --write-occs=$dir/$[$x+1].occs --mix-up=$numgauss --power=$power $dir/$x.mdl \
      "gmm-sum-accs - $dir/$x.*.acc|" $dir/$[$x+1].mdl || exit 1;
    rm $dir/$x.mdl $dir/$x.*.acc $dir/$x.occs 2>/dev/null
  fi
  if [ $x -le $max_iter_inc ]; then
     numgauss=$[$numgauss+$incgauss];
  fi
  beam=10
  x=$[$x+1]
done

( cd $dir; rm final.{mdl,occs} 2>/dev/null; ln -s $x.mdl final.mdl; ln -s $x.occs final.occs )

utils/summarize_warnings.pl $dir/log

echo "Done training tandem mono-phone system in $dir"

# example of showing the alignments:
# show-alignments data/lang/phones.txt $dir/30.mdl "ark:gunzip -c $dir/ali.0.gz|" | head -4

