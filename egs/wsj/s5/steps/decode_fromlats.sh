#!/bin/bash

# Copyright 2012  Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0

# Decode, limited to the word-sequences that were present in a set
# of lattices on disk.  The other lattices do not have to be built
# with the same tree or the same context size-- however, you do
# have to be using the same vocabulary (words.txt)-- if not you'd
# have to map the vocabulary somehow.

# Note: if the trees are identical, you can use gmm-rescore-lattice.

# Mechanism: create an unweighted acceptor (on words) for each utterance,
# compose that with G, determinize, and then use compile-train-graphs-fsts
# to compile a graph for each utterance, to decode with.  

# Begin configuration.
cmd=run.pl
maxactive=7000
beam=20.0
lattice_beam=7.0
acwt=0.083333
batch_size=75 # Limits memory blowup in compile-train-graphs-fsts
scale_opts="--transition-scale=1.0 --self-loop-scale=0.1"
# End configuration.

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;



if [ $# != 4 ]; then
   echo "Usage: steps/decode_si_fromlats.sh [options] <data-dir> <lang> <old-decode-dir> <decode-dir>"
   echo "e.g.: steps/decode_si_fromlats.sh data/test_dev93 data/lang_test_tg exp/tri2b/decode_tgpr_dev93 exp/tri2a/decode_tgpr_dev93_fromlats"
   echo ""
   echo "main options (for others, see top of script file)"
   echo "  --config <config-file>                           # config containing options"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   exit 1;
fi


data=$1
lang=$2
olddir=$3
dir=$4
srcdir=`dirname $dir`; # Assume model directory one level up from decoding directory.

mkdir -p $dir/log

nj=`cat $olddir/num_jobs` || exit 1;
splice_opts=`cat $srcdir/splice_opts 2>/dev/null`
cmvn_opts=`cat $srcdir/cmvn_opts 2>/dev/null`
sdata=$data/split$nj
[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1;
echo $nj >$dir/num_jobs

for f in $sdata/1/feats.scp $sdata/1/cmvn.scp $srcdir/final.mdl $olddir/lat.1.gz \
    $srcdir/tree $lang/L_disambig.fst $lang/phones.txt; do
  [ ! -f $f ] && echo "decode_si_fromlats.sh: no such file $f" && exit 1;
done


if [ -f $srcdir/final.mat ]; then feat_type=lda; else feat_type=delta; fi
echo "decode_si.sh: feature type is $feat_type"

case $feat_type in
  delta) feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | add-deltas ark:- ark:- |";;
  lda) feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | splice-feats $splice_opts ark:- ark:- | transform-feats $srcdir/final.mat ark:- ark:- |";;
  *) echo "Invalid feature type $feat_type" && exit 1;
esac


$cmd JOB=1:$nj $dir/log/decode_lats.JOB.log \
 lattice-to-fst "ark:gunzip -c $olddir/lat.JOB.gz|" ark:- \| \
  fsttablecompose "fstproject --project_output=true $lang/G.fst | fstarcsort |" ark:- ark:- \| \
  fstdeterminizestar ark:- ark:- \| \
  compile-train-graphs-fsts --read-disambig-syms=$lang/phones/disambig.int \
    --batch-size=$batch_size $scale_opts $srcdir/tree $srcdir/final.mdl $lang/L_disambig.fst ark:- ark:- \|  \
  gmm-latgen-faster --max-active=$maxactive --beam=$beam --lattice-beam=$lattice_beam --acoustic-scale=$acwt \
    --allow-partial=true --word-symbol-table=$lang/words.txt \
    $srcdir/final.mdl ark:- "$feats" "ark:|gzip -c > $dir/lat.JOB.gz" || exit 1;

[ ! -x local/score.sh ] && \
  echo "Not scoring because local/score.sh does not exist or not executable." && exit 1;
local/score.sh --cmd "$cmd" $data $lang $dir

exit 0;
