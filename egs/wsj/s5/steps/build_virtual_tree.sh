stage=-4 #  This allows restarting after partway, when something when wrong.
config=
cmd=run.pl
scale_opts="--transition-scale=1.0 --acoustic-scale=0.1 --self-loop-scale=0.1"
# realign_iters="10 20 30"; # no need for this as we don't want to change the alignment
num_iters=35    # Number of iterations of training
max_iter_inc=25 # Last iter to increase #Gauss on.
beam=10
retry_beam=40
boost_silence=1.0 # Factor by which to boost silence likelihoods in alignment
power=0.25 # Exponent for number of gaussians according to occurrence counts
cluster_thresh=0  # for build-tree control final bottom-up clustering of leaves # will back to -1.0
cluster_thresh=-1.0  # for build-tree control final bottom-up clustering of leaves
norm_vars=false # deprecated.  Prefer --cmvn-opts "--norm-vars=true"
                # use the option --cmvn-opts "--norm-means=false"
cmvn_opts=
# End configuration.

numtrees=1

set -e

nj=20
echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh;
. parse_options.sh || exit 1;

if [ $# != 5 ]; then
   echo "Usage: steps/train_virtual_tree.sh <data-dir> <lang-dir> <alignment-dir> <exp-dir>"
   echo "e.g.: steps/train_virtual.sh data/train_si84_half data/lang exp/mono_ali exp/tri3m"
   echo "main options (for others, see top of script file)"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   echo "  --config <config-file>                           # config containing options"
   echo "  --stage <stage>                                  # stage to do partial re-run from."
   exit 1;
fi

data=$1
lang=$2
alidir=$3
dir=$4
outdir=$5



nj=`cat $alidir/num_jobs` || exit 1;
oov=`cat $lang/oov.int` || exit 1;

mkdir -p $outdir
touch $outdir/tree  # not sure why i need this
cp $alidir/splice_opts $outdir
cp $alidir/cmvn_opts $outdir
cp $alidir/final.mat $outdir


if [ $stage -le 3 ]; then
  echo "Now generating the single virtual tree"

  (
  build-tree-virtual --binary=false --num-trees=$numtrees $dir/tree $lang/topo $outdir/tree $outdir/tree-mapping $dir/treeacc || exit 1;) | tee virtual.log

  #../../../src/bin/build-tree-virtual --num-trees=$numtrees exp/tri3m/tree data/lang/topo exp/tri3m/virtual-tree

  echo "generate new models for the virtual tree"

# Note from now, i=virtual !!
  mkdir -p $outdir/log
  cp $dir/tree-* $outdir
  echo $nj > $outdir/num_jobs
  mkdir -p $outdir
  init-model-multi \
    --num-trees=$numtrees \
    $outdir/tree $outdir/tree-mapping $lang/topo $dir/model $outdir/1.mdl 2> $outdir/log/init_model.log || exit 1;
  grep 'no stats' $outdir/log/init_model.log && echo "This is a bad warning.";

#  gmm-mixup --mix-up=$numgauss $dir/tree_$i/1.mdl $dir/tree_$i/1.occs $dir/tree_$i/1.mdl 2>$dir/log/tree_$i/mixup.log || exit 1;
#  rm $dir/treeacc # better keep it here now

  # Convert the alignments.
  echo "$0: converting alignments from $alidir to use virtual tree"
  $cmd JOB=1:$nj $outdir/log/convert.JOB.log \
    convert-ali $alidir/final.mdl $outdir/1.mdl $outdir/tree \
     "ark:gunzip -c $alidir/ali.JOB.gz|" "ark:|gzip -c >$outdir/ali.JOB.gz" || exit 1;
fi

if [ $stage -le 5 ]; then
  echo "$0: compiling graphs of transcripts"
  $cmd JOB=1:$nj $dir/log/compile_graphs.JOB.log \
    compile-train-graphs $outdir/tree $outdir/1.mdl  $lang/L.fst  \
      "ark:utils/sym2int.pl --map-oov $oov -f 2- $lang/words.txt < $data/split$nj/JOB/text |" \
      "ark:|gzip -c >$outdir/fsts.JOB.gz" || exit 1;
fi

cp $outdir/1.mdl $outdir/final.mdl

for i in `seq 0 $[numtrees-1]`; do
  cp $dir/tree_$i/final.mdl $outdir/model-$i
done

