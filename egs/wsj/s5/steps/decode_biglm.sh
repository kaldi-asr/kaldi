#!/usr/bin/env bash

# Copyright 2012  Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0

# Begin configuration.
nj=4
cmd=run.pl
maxactive=7000
beam=13.0
lattice_beam=6.0
acwt=0.083333
skip_scoring=false
# End configuration.

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

if [ $# != 5 ]; then
   echo "Usage: steps/decode_si_biglm.sh [options] <graph-dir> <old-LM-fst> <new-LM-fst> <data-dir> <decode-dir>"
   echo "... where <decode-dir> is assumed to be a sub-directory of the directory"
   echo " where the model is."
   echo "e.g.: steps/decode_si.sh exp/mono/graph_tgpr data/test_dev93 exp/mono/decode_dev93_tgpr"
   echo ""
   echo "This script works on CMN + (delta+delta-delta | LDA+MLLT) features; it works out"
   echo "what type of features you used (assuming it's one of these two)"
   echo ""
   echo "main options (for others, see top of script file)"
   echo "  --config <config-file>                           # config containing options"
   echo "  --nj <nj>                                        # number of parallel jobs"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   exit 1;
fi


graphdir=$1
oldlm_fst=$2
newlm_fst=$3
data=$4
dir=$5

srcdir=`dirname $dir`; # The model directory is one level up from decoding directory.
sdata=$data/split$nj;
splice_opts=`cat $srcdir/splice_opts 2>/dev/null`
cmvn_opts=`cat $srcdir/cmvn_opts 2>/dev/null`
delta_opts=`cat $srcdir/delta_opts 2>/dev/null`

mkdir -p $dir/log
[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1;
echo $nj > $dir/num_jobs


for f in $sdata/1/feats.scp $sdata/1/cmvn.scp $srcdir/final.mdl $graphdir/HCLG.fst $oldlm_fst $newlm_fst; do
  [ ! -f $f ] && echo "decode_si.sh: no such file $f" && exit 1;
done


if [ -f $srcdir/final.mat ]; then feat_type=lda; else feat_type=delta; fi
echo "decode_si.sh: feature type is $feat_type"

case $feat_type in
  delta) feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | add-deltas $delta_opts ark:- ark:- |";;
  lda) feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | splice-feats $splice_opts ark:- ark:- | transform-feats $srcdir/final.mat ark:- ark:- |";;
  *) echo "Invalid feature type $feat_type" && exit 1;
esac

[ -f `dirname $oldlm_fst`/words.txt ] && ! cmp `dirname $oldlm_fst`/words.txt $graphdir/words.txt && \
  echo "Warning: old LM words.txt does not match with that in $graphdir .. probably will not work.";
[ -f `dirname $newlm_fst`/words.txt ] && ! cmp `dirname $oldlm_fst`/words.txt $graphdir/words.txt && \
  echo "Warning: new LM words.txt does not match with that in $graphdir .. probably will not work.";

# fstproject replaces the disambiguation symbol #0, which only appears on the
# input side, with the <eps> that appears in the corresponding arcs on the output side.
oldlm_cmd="fstproject --project_output=true $oldlm_fst | fstarcsort --sort_type=ilabel |"
newlm_cmd="fstproject --project_output=true $newlm_fst | fstarcsort --sort_type=ilabel |"

$cmd JOB=1:$nj $dir/log/decode.JOB.log \
 gmm-latgen-biglm-faster --max-active=$maxactive --beam=$beam --lattice-beam=$lattice_beam \
   --acoustic-scale=$acwt --allow-partial=true --word-symbol-table=$graphdir/words.txt \
  $srcdir/final.mdl $graphdir/HCLG.fst "$oldlm_cmd" "$newlm_cmd" "$feats" \
  "ark:|gzip -c > $dir/lat.JOB.gz" || exit 1;

if ! $skip_scoring ; then
  [ ! -x local/score.sh ] && \
    echo "Not scoring because local/score.sh does not exist or not executable." && exit 1;
  local/score.sh --cmd "$cmd" $data $graphdir $dir ||
    { echo "$0: Scoring failed. (ignore by '--skip-scoring true')"; exit 1; }
fi

exit 0;
