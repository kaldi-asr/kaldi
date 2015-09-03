#!/bin/bash

# Copyright 2012  Johns Hopkins University (Author: Daniel Povey).  Apache 2.0.

# This script does decoding with an SGMM system, by rescoring lattices
# generated from a previous SGMM system.  The directory with the lattices
# is assumed to contain speaker vectors, if used.  Basically it rescores
# the lattices one final time, using the same setup as the final decoding
# pass of the source dir.  The assumption is that the model may have
# been discriminatively trained.

# If the system was built on top of fMLLR transforms from a conventional system,
# you should provide the --transform-dir option.

# Begin configuration section.
transform_dir=    # dir to find fMLLR transforms.
cmd=run.pl
iter=final
# End configuration section.

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

if [ $# -ne 4 ]; then
  echo "Usage: steps/decode_sgmm_rescore.sh [options] <graph-dir|lang-dir> <data-dir> <old-decode-dir> <decode-dir>"
  echo " e.g.: steps/decode_sgmm_rescore.sh --transform-dir exp/tri3b/decode_dev93_tgpr \\"
  echo "      exp/sgmm3a/graph_tgpr data/test_dev93 exp/sgmm3a/decode_dev93_tgpr exp/sgmm3a_mmi/decode_dev93_tgpr"
  echo "main options (for others, see top of script file)"
  echo "  --transform-dir <decoding-dir>           # directory of previous decoding"
  echo "                                           # where we can find transforms for SAT systems."
  echo "  --config <config-file>                   # config containing options"
  echo "  --cmd <cmd>                              # Command to run in parallel with"
  echo "  --iter <iter>                            # iteration of model to use (default: final)"
  exit 1;
fi

graphdir=$1
data=$2
olddir=$3
dir=$4
srcdir=`dirname $dir`; # Assume model directory one level up from decoding directory.

for f in $graphdir/words.txt $data/feats.scp $olddir/lat.1.gz $olddir/gselect.1.gz \
   $srcdir/$iter.mdl; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

nj=`cat $olddir/num_jobs` || exit 1;
sdata=$data/split$nj;
gselect_opt="--gselect=ark,s,cs:gunzip -c $olddir/gselect.JOB.gz|"
splice_opts=`cat $srcdir/splice_opts 2>/dev/null`
cmvn_opts=`cat $srcdir/cmvn_opts 2>/dev/null`

mkdir -p $dir/log
[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1;
echo $nj > $dir/num_jobs

if [ -f $olddir/vecs.1 ]; then
  echo "$0: using speaker vectors from $olddir"
  spkvecs_opt="--spk-vecs=ark:$olddir/vecs.JOB --utt2spk=ark:$sdata/JOB/utt2spk"
else
  echo "$0: no speaker vectors found."
  spkvecs_opt=
fi


## Set up features.
if [ -f $srcdir/final.mat ]; then feat_type=lda; else feat_type=delta; fi
echo "$0: feature type is $feat_type"

case $feat_type in
  delta) feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | add-deltas ark:- ark:- |";;
  lda) feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | splice-feats $splice_opts ark:- ark:- | transform-feats $srcdir/final.mat ark:- ark:- |"
    ;;
  *) echo "$0: invalid feature type $feat_type" && exit 1;
esac
if [ ! -z "$transform_dir" ]; then
  echo "$0: using transforms from $transform_dir"
  [ ! -f $transform_dir/trans.1 ] && echo "$0: no such file $transform_dir/trans.1" && exit 1;
  [ "$nj" -ne "`cat $transform_dir/num_jobs`" ] \
    && echo "$0: #jobs mismatch with transform-dir." && exit 1;
  feats="$feats transform-feats --utt2spk=ark:$sdata/JOB/utt2spk ark,s,cs:$transform_dir/trans.JOB ark:- ark:- |"
elif grep 'transform-feats --utt2spk' $srcdir/log/acc.0.1.log 2>/dev/null; then
  echo "$0: **WARNING**: you seem to be using an SGMM system trained with transforms,"
  echo "  but you are not providing the --transform-dir option in test time."
fi

if [ -f $olddir/trans.1 ]; then
  echo "$0: using (in addition to any previous transforms) transforms from $olddir"
  feats="$feats transform-feats --utt2spk=ark:$sdata/JOB/utt2spk ark,s,cs:$olddir/trans.JOB ark:- ark:- |"
fi
##

# Rescore the state-level lattices with the model provided.  Just
# one command in this script.
echo "$0: rescoring lattices with SGMM model in $srcdir/$iter.mdl"
$cmd JOB=1:$nj $dir/log/rescore.JOB.log \
  sgmm-rescore-lattice "$gselect_opt" $spkvecs_opt \
  $srcdir/$iter.mdl "ark:gunzip -c $olddir/lat.JOB.gz|" "$feats" \
  "ark:|gzip -c > $dir/lat.JOB.gz" || exit 1;

[ ! -x local/score.sh ] && \
  echo "Not scoring because local/score.sh does not exist or not executable." && exit 1;
local/score.sh --cmd "$cmd" $data $graphdir $dir

exit 0;
