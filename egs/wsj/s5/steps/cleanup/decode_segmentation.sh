#!/bin/bash

# Copyright 2014  Guoguo Chen, 2015 GoVivace Inc. (Nagendra Goel)
#           2017  Vimal Manohar
# Apache 2.0

# Some basic error checking, similar to steps/decode.sh is added.

set -e
set -o pipefail

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
parallel_opts=  # ignored now.
scoring_opts=
allow_partial=true
# note: there are no more min-lmwt and max-lmwt options, instead use
# e.g. --scoring-opts "--min-lmwt 1 --max-lmwt 20"
skip_scoring=false
# End configuration section.

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

if [ $# != 3 ]; then
   echo "$0: This is a special decoding script for segmentation where we"
   echo "use one decoding graph per segment. We assume a file HCLG.fsts.scp exists"
   echo "which is the scp file of the graphs for each segment."
   echo "This will normally be obtained by steps/cleanup/make_biased_lm_graphs.sh."
   echo "This script does not estimate fMLLR transforms; you have to use"
   echo "the --transform-dir option if you want to use fMLLR."
   echo ""
   echo "Usage: $0 [options] <graph-dir> <data-dir> <decode-dir>"
   echo " e.g.: $0 exp/tri2b/graph_train_si284_split \\"
   echo "             data/train_si284_split exp/tri2b/decode_train_si284_split"
   echo ""
   echo "where <decode-dir> is assumed to be a sub-directory of the directory"
   echo "where the model is."
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
   exit 1;
fi


graphdir=$1
data=$2
dir=$3

mkdir -p $dir/log

if [ -e $dir/final.mdl ]; then
  srcdir=$dir
elif [ -e $dir/../final.mdl ]; then
  srcdir=$(dirname $dir)
else
  echo "$0: expected either $dir/final.mdl or $dir/../final.mdl to exist"
  exit 1
fi
sdata=$data/split$nj;

[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1;
echo $nj > $dir/num_jobs

if [ -z "$model" ]; then # if --model <mdl> was not specified on the command line...
  if [ -z $iter ]; then model=$srcdir/final.mdl;
  else model=$srcdir/$iter.mdl; fi
fi

if [ $(basename $model) != final.alimdl ] ; then
  # Do not use the $srcpath -- look at the path where the model is
  if [ -f $(dirname $model)/final.alimdl ] && [ -z "$transform_dir" ]; then
    echo -e '\n\n'
    echo $0 'WARNING: Running speaker independent system decoding using a SAT model!'
    echo $0 'WARNING: This is OK if you know what you are doing...'
    echo -e '\n\n'
  fi
fi

for f in $sdata/1/feats.scp $sdata/1/cmvn.scp $model $graphdir/HCLG.fsts.scp; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

utils/lang/check_phones_compatible.sh $graphdir/phones.txt $srcdir/phones.txt

# Split HCLG.fsts.scp by input utterance
n1=$(cat $graphdir/HCLG.fsts.scp | wc -l)
n2=$(cat $data/feats.scp | wc -l)
if [ $n1 != $n2 ]; then
  echo "$0: expected $n2 graphs in $graphdir/HCLG.fsts.scp, got $n1"
fi


mkdir -p $dir/split_fsts
sort -k1,1 $graphdir/HCLG.fsts.scp > $dir/HCLG.fsts.sorted.scp
utils/filter_scps.pl --no-warn -f 1 JOB=1:$nj \
  $sdata/JOB/feats.scp $dir/HCLG.fsts.sorted.scp $dir/split_fsts/HCLG.fsts.JOB.scp
HCLG=scp:$dir/split_fsts/HCLG.fsts.JOB.scp

if [ -f $srcdir/final.mat ]; then feat_type=lda; else feat_type=delta; fi
echo "$0: feature type is $feat_type";

splice_opts=`cat $srcdir/splice_opts 2>/dev/null` || true # frame-splicing options.
cmvn_opts=`cat $srcdir/cmvn_opts 2>/dev/null` || true
delta_opts=`cat $srcdir/delta_opts 2>/dev/null` || true

thread_string=
[ $num_threads -gt 1 ] && thread_string="-parallel --num-threads=$num_threads"

case $feat_type in
  delta) feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | add-deltas $delta_opts ark:- ark:- |";;
  lda) feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | splice-feats $splice_opts ark:- ark:- | transform-feats $srcdir/final.mat ark:- ark:- |";;
  *) echo "Invalid feature type $feat_type" && exit 1;
esac
if [ ! -z "$transform_dir" ]; then # add transforms to features...
  echo "Using fMLLR transforms from $transform_dir"
  [ ! -f $transform_dir/trans.1 ] && echo "Expected $transform_dir/trans.1 to exist" && exit 1
  [ ! -s $transform_dir/num_jobs ] && \
    echo "$0: expected $transform_dir/num_jobs to contain the number of jobs." && exit 1;
  nj_orig=$(cat $transform_dir/num_jobs)
  if [ $nj -ne $nj_orig ]; then
    # Copy the transforms into an archive with an index.
    echo "$0: num-jobs for transforms mismatches, so copying them."
    for n in $(seq $nj_orig); do cat $transform_dir/trans.$n; done | \
       copy-feats ark:- ark,scp:$dir/trans.ark,$dir/trans.scp || exit 1;
    feats="$feats transform-feats --utt2spk=ark:$sdata/JOB/utt2spk scp:$dir/trans.scp ark:- ark:- |"
  else
    # number of jobs matches with alignment dir.
    feats="$feats transform-feats --utt2spk=ark:$sdata/JOB/utt2spk ark:$transform_dir/trans.JOB ark:- ark:- |"
  fi
fi

if [ $stage -le 0 ]; then
  if [ -f "$graphdir/num_pdfs" ]; then
    [ "`cat $graphdir/num_pdfs`" -eq `am-info --print-args=false $model | grep pdfs | awk '{print $NF}'` ] || \
      { echo "Mismatch in number of pdfs with $model"; exit 1; }
  fi
  $cmd --num-threads $num_threads JOB=1:$nj $dir/log/decode.JOB.log \
    gmm-latgen-faster$thread_string --max-active=$max_active --beam=$beam --lattice-beam=$lattice_beam \
    --acoustic-scale=$acwt --allow-partial=$allow_partial --word-symbol-table=$graphdir/words.txt \
    $model "$HCLG" "$feats" "ark:|gzip -c > $dir/lat.JOB.gz" || exit 1;
fi

if ! $skip_scoring ; then
  [ ! -x local/score.sh ] && \
    echo "$0: Not scoring because local/score.sh does not exist or not executable." && exit 1;
  local/score.sh --cmd "$cmd" $scoring_opts $data $graphdir $dir ||
    { echo "$0: Scoring failed. (ignore by '--skip-scoring true')"; exit 1; }
fi

exit 0;
