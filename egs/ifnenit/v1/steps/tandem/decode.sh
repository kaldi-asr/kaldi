#!/bin/bash

# Copyright 2012  Johns Hopkins University (Author: Daniel Povey)
# Apache 2.0

# Begin configuration section.  
transform_dir=
iter=
model= # You can specify the model to use (e.g. if you want to use the .alimdl)
nj=4
cmd=run.pl
max_active=7000
beam=13.0
lattice_beam=6.0
acwt=0.083333 # note: only really affects pruning (scoring is on lattices).
min_lmwt=9
max_lmwt=20
skip_scoring=false
# End configuration section.

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

if [ $# != 4 ]; then
   echo "Usage: steps/tandem/decode.sh [options] <graph-dir> <data1-dir> <data2-dir> <decode-dir>"
   echo "... where <decode-dir> is assumed to be a sub-directory of the directory"
   echo " where the model is."
   echo "e.g.: steps/tandem/decode.sh exp/mono/graph {mfcc,bottleneck}/data/test_dev93 exp/mono/decode_dev93"
   echo ""
   echo "This script works on CMN + (delta+delta-delta | LDA+MLLT) features; it works out"
   echo "what type of features you used (assuming it's one of these two)"
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
   echo "  --min-lmwt <int>                                 # minumum LM-weight for lattice rescoring "
   echo "  --max-lmwt <int>                                 # maximum LM-weight for lattice rescoring "
   echo "                                                   # speaker-adapted decoding"
   exit 1;
fi


graphdir=$1
data1=$2
data2=$3
dir=$4
srcdir=`dirname $dir`; # The model directory is one level up from decoding directory.

mkdir -p $dir/log

sdata1=$data1/split$nj;
sdata2=$data2/split$nj;
[[ -d $sdata1 && $data1/feats.scp -ot $sdata1 ]] || split_data.sh $data1 $nj || exit 1;
[[ -d $sdata2 && $data2/feats.scp -ot $sdata2 ]] || split_data.sh $data2 $nj || exit 1;

echo $nj > $dir/num_jobs

if [ -z "$model" ]; then # if --model <mdl> was not specified on the command line...
  if [ -z $iter ]; then model=$srcdir/final.mdl; 
  else model=$srcdir/$iter.mdl; fi
fi

for f in $sdata1/1/feats.scp $sdata1/1/cmvn.scp $sdata2/1/feats.scp $model $graphdir/HCLG.fst; do
  [ ! -f $f ] && echo "decode.sh: no such file $f" && exit 1;
done

# Set up features.

# Get some info on the feature types
splice_opts=`cat $srcdir/splice_opts 2>/dev/null` # frame-splicing options.
normft2=`cat $srcdir/normft2 2>/dev/null`


if [ -f $srcdir/final.mat ]; then feat_type=lda; else feat_type=delta; fi
echo "decode.sh: feature type is $feat_type";

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
  if [ -e $srcdir/lda.mat ]; then
    feats1="$feats1 splice-feats $splice_opts ark:- ark:- | transform-feats $srcdir/lda.mat ark:- ark:- |"
  else
    feats1="$feats1 add-deltas ark:- ark:- |"
  fi
fi

# set up feature stream 2;  this are usually bottleneck or posterior features, 
# which may be normalized if desired
feats2="scp:$sdata2/JOB/feats.scp"

if [ "$normft2" == "true" ]; then
  echo "Using cmvn for feats2"
  feats2="ark,s,cs:apply-cmvn --norm-vars=false --utt2spk=ark:$sdata2/JOB/utt2spk scp:$sdata2/JOB/cmvn.scp $feats2 ark:- |"
fi

# assemble tandem features
feats="ark,s,cs:paste-feats '$feats1' '$feats2' ark:- |"

# add transformation, if applicable
if [ "$feat_type" == "lda" ]; then
  feats="$feats transform-feats $srcdir/final.mat ark:- ark:- |"
fi

# speaker dependent transformations as requested
if [ ! -z "$transform_dir" ]; then # add transforms to features...
  echo "Using fMLLR transforms from $transform_dir"
  [ ! -f $transform_dir/trans.1 ] && echo "Expected $transform_dir/trans.1 to exist."
  [ "`cat $transform_dir/num_jobs`" -ne $nj ] && \
     echo "Mismatch in number of jobs with $transform_dir";
  feats="$feats transform-feats --utt2spk=ark:$sdata1/JOB/utt2spk ark:$transform_dir/trans.JOB ark:- ark:- |"
fi

$cmd JOB=1:$nj $dir/log/decode.JOB.log \
 gmm-latgen-faster --max-active=$max_active --beam=$beam --lattice-beam=$lattice_beam \
   --acoustic-scale=$acwt --allow-partial=true --word-symbol-table=$graphdir/words.txt \
  $model $graphdir/HCLG.fst "$feats" "ark:|gzip -c > $dir/lat.JOB.gz" || exit 1;

if ! $skip_scoring ; then
  [ ! -x local/score.sh ] && \
    echo "Not scoring because local/score.sh does not exist or not executable." && exit 1;
  local/score.sh --cmd "$cmd" --min_lmwt $min_lmwt --max_lmwt $max_lmwt $data1 $graphdir $dir ||
    { echo "$0: Scoring failed. (ignore by '--skip-scoring true')"; exit 1; }   
fi

exit 0;
