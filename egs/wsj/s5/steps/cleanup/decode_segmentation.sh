#!/bin/bash

# Copyright 2014  Guoguo Chen
# Apache 2.0

# Begin configuration section.  
transform_dir=   # this option won't normally be used, but it can be used if you
                 # want to supply existing fMLLR transforms when decoding.
iter=
model= # You can specify the model to use (e.g. if you want to use the .alimdl)
stage=0
nj=4
cmd=run.pl
max_active=7000
beam=13.0
lattice_beam=6.0
acwt=0.083333 # note: only really affects pruning (scoring is on lattices).
num_threads=1 # if >1, will use gmm-latgen-faster-parallel
parallel_opts=  # If you supply num-threads, you should supply this too.
scoring_opts=
# note: there are no more min-lmwt and max-lmwt options, instead use
# e.g. --scoring-opts "--min-lmwt 1 --max-lmwt 20"
skip_scoring=false
# End configuration section.

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

if [ $# != 3 ]; then
   echo "Usage: $0 [options] <graph-dir> <data-dir> <decode-dir>"
   echo "... where <decode-dir> is assumed to be a sub-directory of the"
   echo "    directory where the model is."
   echo ""
   echo "This is a special decoding script for segmentation where we use one"
   echo "decoding graph for each segment. We assume a file HCLG.fsts.scp exists"
   echo "which is the scp file of the graphs for each segment."
   echo ""
   echo "main options (for others, see top of script file)"
   echo "  --config <config-file>                           # config containing options"
   echo "  --nj <nj>                                        # number of parallel jobs"
   echo "  --iter <iter>                                    # Iteration of model to test."
   echo "  --model <model>                                  # which model to use (e.g. to"
   echo "                                                   # specify the final.alimdl)"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   echo "  --transform-dir <trans-dir>                      # dir to find fMLLR transforms "
   echo "  --acwt <float>                                   # acoustic scale used for lattice generation "
   echo "  --scoring-opts <string>                          # options to local/score.sh"
   echo "  --num-threads <n>                                # number of threads to use, default 1."
   echo "  --parallel-opts <opts>                           # e.g. '-pe smp 4' if you supply --num-threads 4"
   exit 1;
fi


graphdir=$1
data=$2
dir=$3
srcdir=`dirname $dir`; # The model directory is one level up from decoding directory.
sdata=$data/split$nj;

mkdir -p $dir/log
[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1;
echo $nj > $dir/num_jobs

if [ -z "$model" ]; then # if --model <mdl> was not specified on the command line...
  if [ -z $iter ]; then model=$srcdir/final.mdl; 
  else model=$srcdir/$iter.mdl; fi
fi

for f in $sdata/1/feats.scp $sdata/1/cmvn.scp $model $graphdir/HCLG.fsts.scp; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

# Figures out the proper HCLG.
sort -k1,1 -u < $graphdir/HCLG.fsts.scp > $graphdir/HCLG.fsts.scp.sorted
mv $graphdir/HCLG.fsts.scp.sorted $graphdir/HCLG.fsts.scp
for x in `seq 1 $nj`; do
  cat $graphdir/HCLG.fsts.scp |\
    grep -f <(cut -f 1 -d ' ' $sdata/$x/feats.scp) > $sdata/$x/graphs.scp
  num_feats=`cat $sdata/$x/feats.scp | wc -l`
  num_graphs=`cat $sdata/$x/graphs.scp | wc -l`
  if [ $num_graphs -ne $num_feats ]; then
    echo "$0: number of graphs is not consistent with number of features."
    exit 1
  fi
done
HCLG="scp:$sdata/JOB/graphs.scp"

if [ -f $srcdir/final.mat ]; then feat_type=lda; else feat_type=delta; fi
echo "$0: feature type is $feat_type";

splice_opts=`cat $srcdir/splice_opts 2>/dev/null` # frame-splicing options.
cmvn_opts=`cat $srcdir/cmvn_opts 2>/dev/null`
delta_opts=`cat $srcdir/delta_opts 2>/dev/null`

thread_string=
[ $num_threads -gt 1 ] && thread_string="-parallel --num-threads=$num_threads" 

case $feat_type in
  delta) feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | add-deltas $delta_opts ark:- ark:- |";;
  lda) feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | splice-feats $splice_opts ark:- ark:- | transform-feats $srcdir/final.mat ark:- ark:- |";;
  *) echo "Invalid feature type $feat_type" && exit 1;
esac
if [ ! -z "$transform_dir" ]; then # add transforms to features...
  echo "Using fMLLR transforms from $transform_dir"
  [ ! -f $transform_dir/trans.1 ] && echo "Expected $transform_dir/trans.1 to exist."
  [ "`cat $transform_dir/num_jobs`" -ne $nj ] && \
     echo "Mismatch in number of jobs with $transform_dir";
  feats="$feats transform-feats --utt2spk=ark:$sdata/JOB/utt2spk ark:$transform_dir/trans.JOB ark:- ark:- |"
fi

if [ $stage -le 0 ]; then
  if [ -f "$graphdir/num_pdfs" ]; then
    [ "`cat $graphdir/num_pdfs`" -eq `am-info --print-args=false $model | grep pdfs | awk '{print $NF}'` ] || \
      { echo "Mismatch in number of pdfs with $model"; exit 1; }
  fi
  $cmd $parallel_opts JOB=1:$nj $dir/log/decode.JOB.log \
    gmm-latgen-faster$thread_string --max-active=$max_active --beam=$beam --lattice-beam=$lattice_beam \
    --acoustic-scale=$acwt --allow-partial=true --word-symbol-table=$graphdir/words.txt \
    $model "$HCLG" "$feats" "ark:|gzip -c > $dir/lat.JOB.gz" || exit 1;
fi

if ! $skip_scoring ; then
  [ ! -x local/score.sh ] && \
    echo "Not scoring because local/score.sh does not exist or not executable." && exit 1;
  local/score.sh --cmd "$cmd" $scoring_opts $data $graphdir $dir
fi

exit 0;
