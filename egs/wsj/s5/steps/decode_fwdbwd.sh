#!/bin/bash

# Copyright 2012  Johns Hopkins University (Author: Daniel Povey), BUT (Author: Mirko Hannemann)
# Apache 2.0

# Begin configuration section.  
transform_dir=
first_pass=
iter=
model= # You can specify the model to use (e.g. if you want to use the .alimdl)
nj=4
reverse=false
cmd=run.pl
max_active=7000
beam=13.0
lattice_beam=6.0
acwt=0.083333 # note: only really affects pruning (scoring is on lattices).
extra_beam=0.0 # small additional beam over varying beam
max_beam=100.0 # maximum of varying beam
scoring_opts=
# End configuration section.

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

if [ $# != 3 ]; then
   echo "Usage: steps/decode_fwdbwd.sh [options] <graph-dir> <data-dir> <decode-dir>"
   echo "... where <decode-dir> is assumed to be a sub-directory of the directory"
   echo " where the model is."
   echo "e.g.: steps/decode.sh exp/mono/graph_tgpr data/test_dev93 exp/mono/decode_dev93_tgpr"
   echo ""
   echo "This script works on CMN + (delta+delta-delta | LDA+MLLT) features; it works out"
   echo "what type of features you used (assuming it's one of these two)"
   echo ""
   echo "main options (for others, see top of script file)"
   echo "  --config <config-file>                           # config containing options"
   echo "  --first_pass <decode-dir>                        # decoding dir of first pass"
   echo "  --nj <nj>                                        # number of parallel jobs"
   echo "  --iter <iter>                                    # Iteration of model to test."
   echo "  --model <model>                                  # which model to use (e.g. to"
   echo "                                                   # specify the final.alimdl)"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   echo "  --transform_dir <trans-dir>                      # dir to find fMLLR transforms "
   echo "                                                   # speaker-adapted decoding"
   echo "  --scoring-opts <string>                          # options to local/score.sh"
   echo "  --reverse [true/false]                           # time reversal of features"
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

for f in $sdata/1/feats.scp $sdata/1/cmvn.scp $model $graphdir/HCLG.fst $graphdir/words.txt; do
  [ ! -f $f ] && echo "decode_fwdbwd.sh: no such file $f" && exit 1;
done

if [ -f $srcdir/final.mat ]; then feat_type=lda; else feat_type=delta; fi
echo "decode_fwdbwd.sh: feature type is $feat_type";

splice_opts=`cat $srcdir/splice_opts 2>/dev/null`
cmvn_opts=`cat $srcdir/cmvn_opts 2>/dev/null`

case $feat_type in
  delta) feats="ark,s,cs:apply-cmvn $cmvn_opts --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | add-deltas ark:- ark:- |";;
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
if $reverse; then
  feats="$feats reverse-feats ark:- ark:- |"
fi

if [ -f $first_pass/lat.1.gz ]; then
  echo "converting first pass lattice to graph arc acceptor"
  $cmd JOB=1:$nj $dir/log/arc_graph.JOB.log \
    time lattice-arcgraph $model $graphdir/HCLG.fst \
    "ark:gunzip -c $first_pass/lat.JOB.gz|" ark,t:$dir/lat.JOB.arcs || exit 1;
    #  --write-lattices=ark,t:$dir/lat.det
    #  --acoustic-scale=$acwt --lattice-beam=$lattice_beam --prune=false \

  echo "decode with tracking first pass lattice"
  $cmd JOB=1:$nj $dir/log/decode_fwdbwd.JOB.log \
    gmm-latgen-tracking --max-active=$max_active --beam=$beam --lattice-beam=$lattice_beam \
      --acoustic-scale=$acwt --allow-partial=true \
      --extra-beam=$extra_beam --max-beam=$max_beam \
      --word-symbol-table=$graphdir/words.txt  --verbose=2 \
      $model $graphdir/HCLG.fst "$feats" ark:$dir/lat.JOB.arcs "ark:|gzip -c > $dir/lat.JOB.gz" || exit 1;

else
  if [ -f "$graphdir/num_pdfs" ]; then
    [ "`cat $graphdir/num_pdfs`" -eq `am-info --print-args=false $model | grep pdfs | awk '{print $NF}'` ] || \
      { echo "Mismatch in number of pdfs with $model"; exit 1; }
  fi
  $cmd JOB=1:$nj $dir/log/decode.JOB.log \
   gmm-latgen-faster --max-active=$max_active --beam=$beam --lattice-beam=$lattice_beam \
     --acoustic-scale=$acwt --allow-partial=true \
     --word-symbol-table=$graphdir/words.txt \
     $model $graphdir/HCLG.fst "$feats" "ark:|gzip -c > $dir/lat.JOB.gz" || exit 1;
fi

[ ! -x local/score.sh ] && \
  echo "Not scoring because local/score.sh does not exist or not executable." && exit 1;
local/score.sh $scoring_opts --cmd "$cmd" --reverse $reverse $scoring_opts $data $graphdir $dir

echo "Decoding done."
exit 0;
