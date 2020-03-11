#!/usr/bin/env bash

# Copyright 2012  Johns Hopkins University (Author: Daniel Povey)
#                 Korbinian Riedhammer

# Decoding script that does fMLLR.  This can be on top of delta+delta-delta, or
# LDA+MLLT features.

# There are 3 models involved potentially in this script,
# and for a standard, speaker-independent system they will all be the same.
# The "alignment model" is for the 1st-pass decoding and to get the
# Gaussian-level alignments for the "adaptation model" the first time we
# do fMLLR.  The "adaptation model" is used to estimate fMLLR transforms
# and to generate state-level lattices.  The lattices are then rescored
# with the "final model".
#
# The following table explains where we get these 3 models from.
# Note: $srcdir is one level up from the decoding directory.
#
#   Model              Default source:
#
#  "alignment model"   $srcdir/final.alimdl              --alignment-model <model>
#                     (or $srcdir/final.mdl if alimdl absent)
#  "adaptation model"  $srcdir/final.mdl                 --adapt-model <model>
#  "final model"       $srcdir/final.mdl                 --final-model <model>


# Begin configuration section
alignment_model=
adapt_model=
final_model=
transform_dir=
stage=0
acwt=0.083333 # Acoustic weight used in getting fMLLR transforms, and also in
              # lattice generation.
max_active=7000
beam=13.0
lattice_beam=6.0
nj=4
silence_weight=0.01
cmd=run.pl
si_dir=
fmllr_update_type=full
# End configuration section

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

if [ $# != 4 ]; then
   echo "Usage: steps/tandem/mk_aslf_lda_mllt.sh [options] <graph-dir> <data1-dir> <data2-dir> <decode-dir>"
   echo " e.g.: steps/tandem/mk_aslf_lda_mllt.sh exp/tri2b/graph {mfcc,bottleneck}/data/test_dev93 exp/tri2b/decode_dev93"
   echo "main options (for others, see top of script file)"
   echo "  --config <config-file>                   # config containing options"
   echo "  --nj <nj>                                # number of parallel jobs"
   echo "  --cmd <cmd>                              # Command to run in parallel with"
   echo "  --adapt-model <adapt-mdl>                # Model to compute transforms with"
   echo "  --alignment-model <ali-mdl>              # Model to get Gaussian-level alignments for"
   echo "                                           # 1st pass of transform computation."
   echo "  --final-model <finald-mdl>               # Model to finally decode with"
   echo "  --si-dir <speaker-indep-decoding-dir>    # use this to skip 1st pass of decoding"
   echo "                                           # Caution-- must be with same tree"
   echo "  --acwt <acoustic-weight>                 # default 0.08333 ... used to get posteriors"

   exit 1;
fi


graphdir=$1
data1=$2
data2=$3
dir=`echo $4 | sed 's:/$::g'` # remove any trailing slash.

srcdir=`dirname $dir`; # Assume model directory one level up from decoding directory.

mkdir -p $dir/log`

sdata1=$data1/split$nj;
sdata2=$data2/split$nj;
[[ -d $sdata1 && $data1/feats.scp -ot $sdata1 ]] || split_data.sh $data1 $nj || exit 1;
[[ -d $sdata2 && $data2/feats.scp -ot $sdata2 ]] || split_data.sh $data2 $nj || exit 1;

echo $nj > $dir/num_jobs

# Some checks.  Note: we don't need $srcdir/tree but we expect
# it should exist, given the current structure of the scripts.
for f in $graphdir/HCLG.fst $data1/feats.scp $data2/feats.scp $srcdir/tree; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done


## Some checks, and setting of defaults for variables.
[ "$nj" -ne "`cat $dir/num_jobs`" ] && echo "Mismatch in #jobs with si-dir" && exit 1;
[ -z "$adapt_model" ] && adapt_model=$srcdir/final.mdl
[ -z "$final_model" ] && final_model=$srcdir/final.mdl
for f in $adapt_model $final_model; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done
##


# Set up features.

splice_opts=`cat $srcdir/splice_opts 2>/dev/null` # frame-splicing options.
normft2=`cat $srcdir/normft2 2>/dev/null`

if [ -f $srcdir/final.mat ]; then feat_type=lda; else feat_type=delta; fi

case $feat_type in
  delta)
    echo "$0: feature type is $feat_type"
    ;;
  lda)
    echo "$0: feature type is $feat_type"
    ;;
  *) echo "$0: invalid feature type $feat_type" && exit 1;
esac

# set up feature stream 1;  this are usually spectral features, so we will add
# deltas or splice them
feats1="ark,s,cs:apply-cmvn --norm-vars=false --utt2spk=ark:$sdata1/JOB/utt2spk scp:$sdata1/JOB/cmvn.scp scp:$sdata1/JOB/feats.scp ark:- |"

if [ "$feat_type" == "delta" ]; then
  feats1="$feats1 add-deltas ark:- ark:- |"
elif [ "$feat_type" == "lda" ]; then
  feats1="$feats1 splice-feats $splice_opts ark:- ark:- | transform-feats $srcdir/lda.mat ark:- ark:- |"
fi

# set up feature stream 2;  this are usually bottleneck or posterior features,
# which may be normalized if desired
feats2="scp:$sdata2/JOB/feats.scp"

if [ "$normft2" == "true" ]; then
  echo "Using cmvn for feats2"
  feats2="ark,s,cs:apply-cmvn --norm-vars=false --utt2spk=ark:$sdata2/JOB/utt2spk scp:$sdata2/JOB/cmvn.scp $feats2 ark:- |"
fi

# assemble tandem features
sifeats="ark,s,cs:paste-feats '$feats1' '$feats2' ark:- |"

# add transformation, if applicable
if [ "$feat_type" == "lda" ]; then
  sifeats="$sifeats transform-feats $srcdir/final.mat ark:- ark:- |"
fi

if [ -e $dir/trans.1. ]; then
  echo "Using fMLLR transforms in $dir"
  feats="$sifeats transform-feats --utt2spk=ark:$sdata1/JOB/utt2spk ark:$dir/trans.JOB ark:- ark:- |"
elif [ ! -z "$transform_dir" ]; then
  echo "$0: using transforms from $transform_dir"
  [ ! -f $transform_dir/trans.1 ] && echo "$0: no such file $transform_dir/trans.1" && exit 1;
  [ "$nj" -ne "`cat $transform_dir/num_jobs`" ] \
    && echo "$0: #jobs mismatch with transform-dir." && exit 1;
  feats="$sifeats transform-feats --utt2spk=ark:$sdata1/JOB/utt2spk ark,s,cs:$transform_dir/trans.JOB ark:- ark:- |"
elif grep 'transform-feats --utt2spk' $srcdir/log/acc.0.1.log 2>/dev/null; then
  echo "$0: **WARNING**: you seem to be using an SGMM system trained with transforms,"
  echo "  but you are not providing the --transform-dir option in test time."
fi


# Rescore the state-level lattices with the final adapted features, and the final model
# (which by default is $srcdir/final.mdl, but which may be specified on the command line,
# useful in case of discriminatively trained systems).
# At this point we prune and determinize the lattices and write them out, ready for
# language model rescoring.

echo "Rescoring lattices, converting to slf"
mkdir -p $dir/slf
$cmd JOB=1:$nj $dir/log/rescore.slf.JOB.log \
  lattice-align-words $graphdir/phones/word_boundary.int $final_model "ark:gunzip -c $dir/lat.JOB.gz |" ark:- \| \
  gmm-rescore-lattice $final_model ark:- "$feats" ark,t:- \| \
  utils/int2sym.pl -f 3 $graphdir/words.txt \| \
  utils/convert_slf.pl - $dir/slf

exit 0;

